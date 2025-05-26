#ifndef FUSED_ADD_TOPK_DIV_TILING_H
#define FUSED_ADD_TOPK_DIV_TILING_H
#include "register/tilingdata_base.h"

namespace optiling {
BEGIN_TILING_DATA_DEF(FusedAddTopkDivTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, firstDimSize);
  TILING_DATA_FIELD_DEF(uint32_t, secondDimSize);
  TILING_DATA_FIELD_DEF(uint32_t, addNumDimSize);
  TILING_DATA_FIELD_DEF(uint32_t, groupNum);
  TILING_DATA_FIELD_DEF(uint32_t, groupTopk);
  TILING_DATA_FIELD_DEF(uint32_t, n);
  TILING_DATA_FIELD_DEF(uint32_t, k);
  TILING_DATA_FIELD_DEF(uint32_t, activateType);
  TILING_DATA_FIELD_DEF(uint32_t, isNorm);
  TILING_DATA_FIELD_DEF(float, scale);
  TILING_DATA_FIELD_DEF(uint32_t, groupEles);
  TILING_DATA_FIELD_DEF(uint32_t, blockNum);
  TILING_DATA_FIELD_DEF(uint32_t, ubFactorElement);
  TILING_DATA_FIELD_DEF(uint32_t, batchPerCore);
  TILING_DATA_FIELD_DEF(uint32_t, tailBatch);
  TILING_DATA_FIELD_DEF(uint32_t, tilingKey);
  TILING_DATA_FIELD_DEF(uint32_t, dtype);
  TILING_DATA_FIELD_DEF(uint32_t, tempSize);
  TILING_DATA_FIELD_DEF(uint64_t, workspacePerCore);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(FusedAddTopkDiv, FusedAddTopkDivTilingData)
}
#endif // FUSED_ADD_TOPK_DIV_TILING_H
