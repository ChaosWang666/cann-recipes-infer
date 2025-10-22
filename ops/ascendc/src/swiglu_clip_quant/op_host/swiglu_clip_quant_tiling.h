/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file swiglu_clip_quant_tiling.h
 * \brief
 */

#ifndef SWIGLU_CLIP_QUANT_TILING_H
#define SWIGLU_CLIP_QUANT_TILING_H


#include <vector>
#include <iostream>
#include "register/op_impl_registry.h"
#include "platform/platform_infos_def.h"
#include "swiglu_clip_quant_def.cpp"
#include "exe_graph/runtime/tiling_context.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/tiling_api.h"
#include "error/ops_error.h"
#include "platform/platform_info.h"

namespace optiling {
// ----------公共定义----------
struct TilingRequiredParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::StorageShape *shape;
};

struct TilingOptionalParaInfo {
    const gert::CompileTimeTensorDesc *desc;
    const gert::Tensor *tensor;
};
// ----------算子TilingData定义----------
BEGIN_TILING_DATA_DEF(SwigluClipQuantTilingData)
TILING_DATA_FIELD_DEF(int64_t, inDimx);
TILING_DATA_FIELD_DEF(int64_t, inDimy);
TILING_DATA_FIELD_DEF(int64_t, outDimy);
TILING_DATA_FIELD_DEF(int64_t, UbFactorDimx);
TILING_DATA_FIELD_DEF(int64_t, UbFactorDimy);  // cut for output dim
TILING_DATA_FIELD_DEF(int64_t, usedCoreNum);
TILING_DATA_FIELD_DEF(int64_t, maxCoreNum);
TILING_DATA_FIELD_DEF(int64_t, inGroupNum);
TILING_DATA_FIELD_DEF(int64_t, hasBias);
TILING_DATA_FIELD_DEF(int64_t, quantMode);
TILING_DATA_FIELD_DEF(int64_t, actRight);
TILING_DATA_FIELD_DEF(int64_t, quantScaleDtype);
TILING_DATA_FIELD_DEF(int64_t, groupIndexDtype);
TILING_DATA_FIELD_DEF(int64_t, needSmoothScale);
TILING_DATA_FIELD_DEF(int64_t, biasDtype);
TILING_DATA_FIELD_DEF(int64_t, speGroupType);
TILING_DATA_FIELD_DEF(int64_t, activationScaleIsEmpty);
TILING_DATA_FIELD_DEF(int64_t, quantIsOne);
TILING_DATA_FIELD_DEF(int64_t, clampMode);
TILING_DATA_FIELD_DEF(int64_t, hasGroupAlpha);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(SwigluClipQuant, SwigluClipQuantTilingData)

// ----------算子CompileInfo定义----------
struct SwigluClipQuantCompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSize = 0;
};

// ----------算子Tiling入参信息解析及check类----------
class SwigluClipQuantAllTiling {
public:
    explicit SwigluClipQuantAllTiling(gert::TilingContext* tilingContext) : context_(tilingContext)
        {
        }
    ~SwigluClipQuantAllTiling() = default;

    uint64_t coreNum_ = 0;
    uint64_t ubSize_ = 0;
    int64_t groupNum_ = 0;
    int64_t actRight_ = 0;
    int64_t quantMode_ = 0;
    uint64_t workspaceSize_ = 0;
    int64_t maxPreCore_ = 0;
    bool hasWeightScale_ = false;
    bool hasActivationScale_ = false;
    bool hasBias_ = false;
    bool hasQuantScale_ = false;
    bool hasQuantOffset_ = false;
    bool hasGroupIndex_ = false;
    bool speGroupType_ = false;
    bool hasGroupAlpha_ = false;

    // variable for SwiGLU used by GPT-OSS
    int64_t clampMode_ = 0;
    float clampLimit_ = 0.0;
    float gluAlpha_ = 0.0;
    float gluBias_ = 0.0;
    
    ge::graphStatus GetPlatformInfo();
    ge::graphStatus DoOpTiling();
    uint64_t GetTilingKey() const;
    ge::graphStatus GetWorkspaceSize();
    ge::graphStatus PostTiling();
    void DumpTilingInfo();
    ge::graphStatus GetAttr();
    ge::graphStatus CheckXAndGroupIndexDtype();
    void CountTilingKey();
    ge::graphStatus CountMaxDim(int64_t& ubFactorDimx);
    bool IsPerformanceAndGroupIndexBrach();
    ge::graphStatus GetShapeAttrsInfoInner();
    static bool CheckOptionalShapeExisting(const gert::StorageShape* storageShape);

private:
    gert::TilingContext *context_ = nullptr;
    uint64_t tilingKey_ = 0;
    SwigluClipQuantTilingData tilingData_;
    int64_t inDimx_ = 0;
    int64_t inDimy_ = 0;
    int64_t outDimy_ = 0;
    platform_ascendc::SocVersion socVersion = platform_ascendc::SocVersion::ASCEND910B;
};

// ----------算子Tiling类----------
class SwigluClipQuantTiling {
public:
    explicit SwigluClipQuantTiling(gert::TilingContext *tilingContext) : context_(tilingContext){};
    ge::graphStatus DoTiling(SwigluClipQuantAllTiling *tilingInfo);

private:
    gert::TilingContext *context_ = nullptr;
    SwigluClipQuantTilingData tilingData_;
};

}  // namespace optiling
#endif  // SWIGLU_CLIP_QUANT_TILING_H
