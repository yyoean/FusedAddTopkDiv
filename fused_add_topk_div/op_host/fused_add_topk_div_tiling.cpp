#include "fused_add_topk_div_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{

  FusedAddTopkDivTilingData tiling;
  /*
  const gert::StorageShape* x1_shape = context->GetInputShape(0);
  int32_t data_sz = 1;
  for (int i = 0; i < x1_shape->GetStorageShape().GetDimNum(); i++)
    data_sz *= x1_shape->GetStorageShape().GetDim(i);
  tiling.set_size(data_sz);
  */
  auto firstDimSize = context->GetInputShape(0)->GetStorageShape().GetDim(0);
  auto secondDimSize = context->GetInputShape(0)->GetStorageShape().GetDim(1);
  auto addNumDimSize = context->GetInputShape(1)->GetStorageShape().GetDim(0);

  int group_num = *context->GetAttrs()->GetInt(0);
  int group_topk = *context->GetAttrs()->GetInt(1);
  int n = *context->GetAttrs()->GetInt(2);
  int k = *context->GetAttrs()->GetInt(3);
  int activate_type = *context->GetAttrs()->GetInt(4);
  auto is_norm = *context->GetAttrs()->GetBool(5);
  auto scale = *context->GetAttrs()->GetFloat(6);

  auto groupEles = group_num == 0 ? secondDimSize : secondDimSize / group_num;
  auto dtype = context->GetInputDesc(0)->GetDataType();
  uint64_t ubSize = 0;
  auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
  ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
  ubSize = ubSize - 16 * 1024;
  ubSize = ubSize / 32 * 32;

  auto coreNum = ascendcPlatform.GetCoreNumAiv();

  auto usedCoreNum = 0;
  auto batchPerCore = 0;
  auto tailBatch = 0;
  if (firstDimSize <= coreNum){
      batchPerCore = 1;
      usedCoreNum = firstDimSize;
      tailBatch = 0;
  }else {
      batchPerCore = coreNum == 0 ? firstDimSize : firstDimSize / coreNum;
      tailBatch = firstDimSize % coreNum;
      usedCoreNum = coreNum;
  }
  auto blockNum = usedCoreNum;

  uint32_t tilingDataSize = (tiling.GetDataSize() + 31) / 32 * 32;
  uint32_t canUseUbSize = (ubSize-tilingDataSize) / 32 * 32;
  auto ubFactorElement = canUseUbSize / 32 * 32;

  auto tempSize = firstDimSize * secondDimSize * 4;
  auto workspacePerCore = 256 * 4;

  uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
  size_t *currentWorkspace = context->GetWorkspaceSizes(1);
  auto usrSize = blockNum * workspacePerCore;
  currentWorkspace[0] = usrSize + sysWorkspaceSize;

  tiling.set_firstDimSize(firstDimSize);
  tiling.set_secondDimSize(secondDimSize);
  tiling.set_addNumDimSize(addNumDimSize);
  tiling.set_groupNum(group_num);
  tiling.set_groupTopk(group_topk);
  tiling.set_n(n);
  tiling.set_k(k);
  tiling.set_activateType(activate_type);
  tiling.set_isNorm(is_norm);
  tiling.set_scale(scale);
  tiling.set_groupEles(groupEles);
  tiling.set_dtype(dtype);
  tiling.set_blockNum(blockNum);
  tiling.set_batchPerCore(batchPerCore);
  tiling.set_tailBatch(tailBatch);
  tiling.set_tilingKey(0);
  tiling.set_ubFactorElement(ubFactorElement);
  tiling.set_tempSize(tempSize);
  tiling.set_workspacePerCore(workspacePerCore);

  context->SetBlockDim(usedCoreNum);
  tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
  context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
  context->SetTilingKey(0);

    printf("Tiling Details:\n");
    printf("First Dimension Size: %ld\n", tiling.get_firstDimSize());
    printf("Second Dimension Size: %ld\n", tiling.get_secondDimSize());
    printf("Add Num Dimension Size: %ld\n", tiling.get_addNumDimSize());
    printf("Group Number: %ld\n", tiling.get_groupNum());
    printf("Group Topk: %ld\n", tiling.get_groupTopk());
    printf("n: %ld\n", tiling.get_n());
    printf("k: %ld\n", tiling.get_k());
    printf("Activate Type: %ld\n", tiling.get_activateType());
    printf("Is Norm: %d\n", tiling.get_isNorm());
    printf("Scale: %f\n", tiling.get_scale());
    printf("Group Elements: %ld\n", tiling.get_groupEles());
    printf("Data Type: %ld\n", tiling.get_dtype());
    printf("Block Number: %ld\n", tiling.get_blockNum());
    printf("Batch Per Core: %ld\n", tiling.get_batchPerCore());
    printf("Tail Batch: %ld\n", tiling.get_tailBatch());
    printf("Tiling Key: %ld\n", tiling.get_tilingKey());
    printf("UB Factor Element: %ld\n", tiling.get_ubFactorElement());
    printf("Temp Size: %ld\n", tiling.get_tempSize());
    printf("Workspace Per Core: %ld\n", tiling.get_workspacePerCore());
  return ge::GRAPH_SUCCESS;
}
}


namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    //gert::Shape out1 = gert::Shape({16,8});
    int firstDimSize = context->GetInputShape(0)->GetDim(0);
    int k = *context->GetAttrs()->GetInt(3);
    gert::Shape out1 = gert::Shape({firstDimSize,k});
    //printf("%d%d",firstDimSize,k);
    gert::Shape* outTensor1_shape = context->GetOutputShape(0);
    gert::Shape* outTensor2_shape = context->GetOutputShape(1);
    *outTensor1_shape = {firstDimSize,k};
    *outTensor2_shape = {firstDimSize,k};
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext *context)
{
    const auto inputDataType = context->GetInputDataType(0);
    //const auto inputDataType = context->GetInputDataType(1);

    context->SetOutputDataType(0, ge::DT_FLOAT);
    context->SetOutputDataType(1, ge::DT_INT32);
    return ge::GRAPH_SUCCESS;
}
}


namespace ops {
class FusedAddTopkDiv : public OpDef {
public:
    explicit FusedAddTopkDiv(const char* name) : OpDef(name)
    {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
        this->Input("add_num")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT16, ge::DT_BF16, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

	this->Attr("group_num").AttrType(REQUIRED).Int();
	this->Attr("group_topk").AttrType(REQUIRED).Int();
	this->Attr("n").AttrType(REQUIRED).Int();
	this->Attr("k").AttrType(REQUIRED).Int();
	this->Attr("activate_type").AttrType(REQUIRED).Int();
	this->Attr("is_norm").AttrType(REQUIRED).Bool();
	this->Attr("scale").AttrType(OPTIONAL).Float(1.0);


        this->Output("y")
            .ParamType(REQUIRED)
            .DataType({ge::DT_FLOAT, ge::DT_FLOAT, ge::DT_FLOAT})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});
	this->Output("indices")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32, ge::DT_INT32, ge::DT_INT32})
            .Format({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND, ge::FORMAT_ND, ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910b");

    }
};

OP_ADD(FusedAddTopkDiv);
}
