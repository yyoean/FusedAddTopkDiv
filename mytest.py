import torch
import torch_npu
from typing import Callable, Optional
import numpy as np

torch.manual_seed(42)
input_x = torch.rand([16, 256], dtype = torch.float32).npu()
input_addNum = torch.rand([256], dtype = torch.float32).npu()
#print("========input_x:========")
#print(input_x,input_x.shape)
#print("========input_add_num========")
#print(input_addNum, input_addNum.shape)

output_y, outpt_indices = torch_npu.fused_add_topk_div(
        x = input_x,
        add_num = input_addNum,
        group_num = 8,
        group_topk = 4,
        n = 2,
        k = 8,
        activate_type = 0,
        is_norm = True,
        scale = 1.0
        )
#print("output_y:",output_y)
#print("outpt_indices",outpt_indices)

'''
output_y1, outpt_indices1 = torch_npu.npu_moe_gating_top_k(
        x = input_x,
        add_num = input_addNum,
        group_num = 8,
        group_topk = 4,
        n = 2,
        k = 8,
        activate_type = 0,
        is_norm = True,
        scale = 1.0
        )
'''

def golden(input_x, input_add_num, group_num, group_topk, n, k, activate_type, is_norm, scale):
    input_x = input_x.to(torch.float32).npu()
    input_add_num = input_add_num.to(torch.float32).npu()
    #sigmod
    m = torch.nn.Sigmoid()
    output_sig = m(input_x)
    #print("========sigmoid output_sig========")
    #print(output_sig)

    #add
    input0 = torch.add(output_sig, input_add_num)
    #print("========add input0========")
    #print(input0)

    #group_topk
    token_num, expert_num = input0.shape
    #print("========token_num========")
    #print(token_num)
    #print("========expert_num========")
    #print(expert_num)
    input0 = torch.reshape(input0, (token_num, group_num, expert_num // group_num))
    #print("========input0========")
    #print(input0.shape)
    output = input0.clone()
    
    group_tensor = torch.topk(input0, n).values
    #print("========group_tensor========")
    #print(group_tensor, group_tensor.shape)

    group_tensor = torch.sum(group_tensor, dim = -1)
    #print("========group_tensor========")
    #print(group_tensor, group_tensor.shape)

    sort_index = torch.from_numpy(np.argsort(-group_tensor.cpu().numpy(), kind = 'stable'))
    #print("========sort_index========")
    #print(sort_index,sort_index.shape)
    cols_to_use = torch.arange(group_topk, group_num, dtype = torch.long)
    #print("========cols_to_use========")
    #print(cols_to_use)
    row_indices = torch.arange(sort_index.shape[0]).repeat_interleave(cols_to_use.shape[0])
    #print("========row_indices========")
    #print(row_indices)
    col_indices = sort_index.index_select(1, cols_to_use).view(-1)
    #print("========col_indices========")
    #print(col_indices)
    output[row_indices, col_indices] = float(0)
    #print("========output========")
    #print(output,output.shape)
    group_top_k_res = torch.reshape(output, (token_num, expert_num))
    #print("========group_top_k_res========")
    #print(group_top_k_res, group_top_k_res.shape)
    #for i in range(group_top_k_res.shape[0]):
    #    print(f"Row {i}: {group_top_k_res[i, :]}")
    #print("shape",group_top_k_res.shape)
    
    #topk
    sort_res = torch.sort(group_top_k_res, descending = True, stable = True)
    #print("========sort_res========")
    #print(sort_res)
    #gather
    gather_res = torch.gather(output_sig, -1, sort_res.indices[:,0:k])
    #print("========gather_res========")
    #print(gather_res,gather_res.shape)
    if is_norm:
        #reduce sum
        sum_res = torch.sum(gather_res, -1, keepdim = True)
        #div
        res = torch.div(gather_res, sum_res)
        #mul
        res = res * torch.tensor(scale, dtype = torch.float32)
    else:
        res = gather_res
    #print("========res========")
    #print(res, res.shape)
    #print("========sort_res========")
    #temp = sort_res.indices[:,0:k].to(torch.int32)
    #print(temp,temp.shape)
    return res, sort_res.indices[:,0:k].to(torch.int32)

golden_y, golden_indices = golden(
        input_x = input_x,
        input_add_num = input_addNum,
        group_num = 8,
        group_topk = 4,
        n = 1,
        k = 8,
        activate_type = 0,
        is_norm = True,
        scale = 1.0
        )
print("=============golden===========")
print("golden_y:", golden_y)
print("golden_indices:",golden_indices)

def native_grouped_topk(
    topk_weights: torch.Tensor,
    num_expert_group: Optional[int],
    topk_group: Optional[int],
):
    #print("initial===============topk_weights",topk_weights, topk_weights.shape)
    topk_group = 0 if topk_group is None else topk_group
    num_expert_group = 0 if num_expert_group is None else num_expert_group

    num_token = topk_weights.shape[0]
    #print("==========num_token",num_token)
    #print("==========num_expert_group",num_expert_group)
    grouped_weights = topk_weights.view(num_token, num_expert_group,
                                        -1).max(dim=-1).values
    #print("before======================grouped_weights",grouped_weights, grouped_weights.shape,"topk_group",topk_group)
    topk_group_indices = torch.topk(grouped_weights.to(torch.float32),
                                    k=topk_group,
                                    dim=-1,
                                    sorted=False)[1]
    #print("after======================grouped_weights",grouped_weights.to(torch.float32), grouped_weights.shape)
    #print("topk_group_indices",topk_group_indices,topk_group_indices.shape)
    topk_group_mask = torch.zeros_like(grouped_weights)

    topk_group_mask.scatter_(1, topk_group_indices, 1)
    topk_weight_mask = (topk_group_mask.unsqueeze(-1).expand(
        num_token, num_expert_group,
        topk_weights.shape[-1] // num_expert_group).reshape(num_token, -1))
    topk_weights = topk_weights.masked_fill(~topk_weight_mask.bool(), 0.0)

    return topk_weights


def select_experts(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    top_k: int,
    use_grouped_topk: bool,
    renormalize: bool,
    topk_group: Optional[int] = None,
    num_expert_group: Optional[int] = None,
    custom_routing_function: Optional[Callable] = None,
    scoring_func: str = "softmax",
    e_score_correction_bias: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Select top-k experts based on router logits.

    Args:
        hidden_states: Hidden states of shape (num_tokens, hidden_size).
        router_logits: Router logits of shape (num_tokens, num_experts).
        top_k: Number of experts to select.
        use_grouped_topk: Whether to group experts before selecting top-k.
        renormalize: Whether to renormalize the routing weights.
        topk_group: Number of expert groups to select from.
        num_expert_group: Number of experts in each group.
        custom_routing_function: Custom routing function.
        scoring_func: Scoring function to use.
        e_score_correction_bias: Correction bias to apply to expert scores.

    Returns:
        topk_weights: Routing weights of shape (num_tokens, top_k).
        topk_ids: Selected expert IDs of shape (num_tokens, top_k).

    Raises:
        ValueError: If an unsupported scoring function is provided.
    """

    if scoring_func == "softmax":
        # NOTE: vLLM use dtype=torch.float here
        topk_weights = router_logits.softmax(dim=-1)
    elif scoring_func == "sigmoid":
        topk_weights = router_logits.sigmoid()
        #print("sigmoid",topk_weights)
    else:
        raise ValueError(f"Unsupported scoring function: {scoring_func}")

    if use_grouped_topk:
        assert topk_group is not None
        assert num_expert_group is not None

        if e_score_correction_bias is not None:
            # Store original scores before applying correction bias. We use biased
            # scores for expert selection but original scores for routing weights
            original_weights = topk_weights
            topk_weights = topk_weights + e_score_correction_bias.unsqueeze(0)
            #print("add",topk_weights,topk_weights.shape)

        # TODO: Change to npu_group_topk when the latest CANN and NNAL is available
        # >>> torch_npu._npu_group_topk(topk_weights, group_num=num_expert_group, k=topk_group)
        topk_weights = native_grouped_topk(topk_weights, num_expert_group, topk_group)
        #print("group_topk",topk_weights, topk_weights.shape)
        #for i in range(topk_weights.shape[0]):
        #    print(f"Row {i}: {topk_weights[i, :]}")
        #print("shape",topk_weights.shape)

        # TODO bfloat16 is not supported in torch.topk with ge graph.
        if e_score_correction_bias is not None:
            topk_ids = torch.topk(topk_weights.to(torch.float32),
                                  k=top_k,
                                  dim=-1,
                                  sorted=False)[1]
            #print("topk",topk_ids)
            # Use original unbiased scores for the routing weights
            topk_weights = original_weights.gather(1, topk_ids)
            #print("gather",topk_weights)
        else:
            topk_weights, topk_ids = torch.topk(topk_weights.to(torch.float32),
                                                k=top_k,
                                                dim=-1,
                                                sorted=False)
    else:
        topk_weights, topk_ids = topk_weights.topk(top_k, dim=-1)
        topk_weights = topk_weights.to(hidden_states.dtype)

    # Required by npu_moe_init_routing
    topk_ids = topk_ids.to(torch.int32)

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights, topk_ids
         

topk_weights, topk_ids = select_experts(
    hidden_states=input_addNum,
    router_logits=input_x,
    top_k=8, #k
    use_grouped_topk=True,
    renormalize=True,
    topk_group=4, #group_topk or n 8?
    num_expert_group=8, #group_num
    custom_routing_function=None,
    scoring_func="sigmoid",
    e_score_correction_bias=input_addNum,
    )
print("==========select_experts===========")
print("topk_weights:",topk_weights)
print("topk_ids:",topk_ids)

topk_weights, topk_ids, _ = torch_npu.npu_moe_gating_top_k(
    x=input_x,
    k=8,  # topk�~S�~I~M�~F~Y8
    bias=input_addNum,
    k_group=4,  # fix: 4
    group_count=8,  # fix 8
    group_select_mode=1,  # 0: group中�~Z~D�~\~@大; 1: topk2.sum(fix)
    renorm=0,  # 0: softmax->topk(fix); 1: topk->softmax
    norm_type=1,  # 0: softmax; 1: sigmoid(fix)
                # out_flag=False, # todo new api; 第�~I个�~S�~G��~X��~P��~S�~G�
                # y2_flag=False, # old api; 第�~I个�~S�~G��~X��~P��~S�~G�
    routed_scaling_factor=1,
    eps=float(1e-20))
print("============npu_moe_gating_top_k==========")
print("topk_weights",topk_weights)
print("topk_ids",topk_ids)
