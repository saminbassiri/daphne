/*
 * Copyright 2021 The DAPHNE Consortium
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef SRC_RUNTIME_LOCAL_KERNELS_GROUP_H
#define SRC_RUNTIME_LOCAL_KERNELS_GROUP_H

#include <ir/daphneir/Daphne.h>
#include <runtime/local/context/DaphneContext.h>
#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>
#include <runtime/local/datastructures/Frame.h>
#include <runtime/local/datastructures/ValueTypeCode.h>
#include <runtime/local/datastructures/ValueTypeUtils.h>
#include <runtime/local/kernels/CastSca.h>
#include <runtime/local/kernels/ExtractCol.h>
#include <runtime/local/kernels/Order.h>
#include <util/DeduceType.h>

#include <iterator>
#include <vector>

// ****************************************************************************
// Struct for partial template specialization
// ****************************************************************************

template <class DT> struct Group {
    static void apply(DT *&res, const DT *arg, const char **keyCols, size_t numKeyCols, const char **aggCols,
                      size_t numAggCols, mlir::daphne::GroupEnum *aggFuncs, size_t numAggFuncs, DCTX(ctx)) = delete;
};

// ****************************************************************************
// Convenience function
// ****************************************************************************

template <class DT>
void group(DT *&res, const DT *arg, const char **keyCols, size_t numKeyCols, const char **aggCols, size_t numAggCols,
           mlir::daphne::GroupEnum *aggFuncs, size_t numAggFuncs, DCTX(ctx)) {
    Group<DT>::apply(res, arg, keyCols, numKeyCols, aggCols, numAggCols, aggFuncs, numAggFuncs, ctx);
}

// ****************************************************************************
// (Partial) template specializations for different data/value types
// ****************************************************************************

// ----------------------------------------------------------------------------
// Frame <- Frame
// ----------------------------------------------------------------------------

// returns the result of the aggregation function aggFunc over the (contiguous)
// memory between the begin and end pointer
template <typename VTRes, typename VTArg>
VTRes aggregate(const mlir::daphne::GroupEnum &aggFunc, const VTArg *begin, const VTArg *end) {
    using mlir::daphne::GroupEnum;
    switch (aggFunc) {
    case GroupEnum::COUNT:
        return end - begin;
        break; // TODO: Do we need to check for Null elements here?
    case GroupEnum::SUM:
        return std::accumulate(begin, end, (VTRes)0);
        break;
    case GroupEnum::MIN:
        return *std::min_element(begin, end);
        break;
    case GroupEnum::MAX:
        return *std::max_element(begin, end);
        break;
    case GroupEnum::AVG:
        return std::accumulate(begin, end, (double)0) / (double)(end - begin);
        break;
    default:
        return *begin;
        break;
    }
}

// struct which calls the aggregate() function (specified via aggFunc) on each
// duplicate group in the groups vector and on all implied single groups for a
// sepcified column (colIdx) of the argument frame (arg) and stores the result
// in the specified column (colIdx) of the result frame (res)
template <typename VTRes, typename VTArg> struct ColumnGroupAgg {
    static void apply(Frame *res, const Frame *arg, size_t colIdx, std::vector<std::pair<size_t, size_t>> *groups,
                      mlir::daphne::GroupEnum aggFunc, DCTX(ctx)) {
        VTRes *valuesRes = res->getColumn<VTRes>(colIdx)->getValues();
        const VTArg *valuesArg = arg->getColumn<VTArg>(colIdx)->getValues();
        size_t rowRes = 0;
        size_t numRows = arg->getNumRows();

        // case for no duplicates
        if (groups == nullptr || groups->empty()) {
            for (size_t r = 0; r < numRows; r++)
                valuesRes[rowRes++] = aggregate<VTRes, VTArg>(aggFunc, valuesArg + r, valuesArg + r + 1);
            return;
        }

        for (size_t r = 0; r < groups->front().first; r++)
            valuesRes[rowRes++] = aggregate<VTRes, VTArg>(aggFunc, valuesArg + r, valuesArg + r + 1);
        for (auto it = groups->begin(); it != groups->end(); ++it) {
            valuesRes[rowRes++] = aggregate<VTRes, VTArg>(aggFunc, valuesArg + it->first, valuesArg + it->second);
            for (size_t r = it->second; r < (std::next(it) != groups->end() ? std::next(it)->first : it->second); r++) {
                valuesRes[rowRes++] = aggregate<VTRes, VTArg>(aggFunc, valuesArg + r, valuesArg + r + 1);
            }
        }
        for (size_t r = groups->back().second; r < numRows; r++)
            valuesRes[rowRes++] = aggregate<VTRes, VTArg>(aggFunc, valuesArg + r, valuesArg + r + 1);
    }
};

inline std::string myStringifyGroupEnum(mlir::daphne::GroupEnum val) {
    using mlir::daphne::GroupEnum;
    switch (val) {
    case GroupEnum::COUNT:
        return "COUNT";
    case GroupEnum::SUM:
        return "SUM";
    case GroupEnum::MIN:
        return "MIN";
    case GroupEnum::MAX:
        return "MAX";
    case GroupEnum::AVG:
        return "AVG";
    }
    return "";
}

template <> struct Group<Frame> {
    static void apply(Frame *&res, const Frame *arg, const char **keyCols, size_t numKeyCols, const char **aggCols,
                      size_t numAggCols, mlir::daphne::GroupEnum *aggFuncs, size_t numAggFuncs, DCTX(ctx)) {
        size_t numRowsArg = arg->getNumRows();
        size_t numColsRes = numKeyCols + numAggCols;
        size_t numRowsRes = numRowsArg;
        if (arg == nullptr || (keyCols == nullptr && numKeyCols != 0) || (aggCols == nullptr && numAggCols != 0) ||
            (aggFuncs == nullptr && numAggFuncs != 0)) {
            throw std::runtime_error("group-kernel called with invalid arguments");
        }

        // check if labels contain *
        std::vector<std::string> starLabels;
        const std::string *argLabels = arg->getLabels();
        const size_t numColsArg = arg->getNumCols();
        std::vector<std::string> aggColsVec;
        for (size_t m = 0; m < numAggCols; m++) {
            aggColsVec.push_back(aggCols[m]);
        }
        for (size_t i = 0; i < numKeyCols; i++) {
            std::string delimiter = ".";
            std::string keyLabel = keyCols[i];
            const std::string frameName = keyLabel.substr(0, keyLabel.find(delimiter));
            const std::string colLabel =
                keyLabel.substr(keyLabel.find(delimiter) + delimiter.length(), keyLabel.length());
            if (strcmp(keyCols[i], "*") == 0) {
                for (size_t m = 0; m < numColsArg; m++) {
                    // check that we do not include columns in the result that
                    // are used for aggregations and would lead to duplicates
                    if (std::find(aggColsVec.begin(), aggColsVec.end(), argLabels[m]) == aggColsVec.end()) {
                        starLabels.push_back(argLabels[m]);
                    }
                }
                // we assume that other key columns are included in the *
                // operator, otherwise they would not be in the argument frame
                // and throw a error later on
                numColsRes = starLabels.size() + numAggCols;
            } else if (colLabel.compare("*") == 0) { // f.*
                for (size_t m = 0; m < numColsArg; m++) {
                    std::string frameArg = argLabels[m].substr(0, argLabels[m].find(delimiter));
                    if (frameName.compare(argLabels[m].substr(0, argLabels[m].find(delimiter))) == 0 &&
                        frameName.compare(frameArg) == 0) {
                        starLabels.push_back(argLabels[m]);
                    }
                }
                numColsRes = starLabels.size() + numAggCols;
            }
        }

        // convert labels to indices
        auto idxs = std::shared_ptr<size_t[]>(new size_t[numColsRes]);
        numKeyCols = starLabels.size() ? starLabels.size() : numKeyCols;
        bool *ascending = new bool[starLabels.size()];
        for (size_t i = 0; i < numKeyCols; ++i) {
            idxs[i] = starLabels.size() ? arg->getColumnIdx(starLabels[i]) : arg->getColumnIdx(keyCols[i]);
            ascending[i] = true;
        }
        for (size_t i = numKeyCols; i < numColsRes; i++) {
            idxs[i] = arg->getColumnIdx(aggCols[i - numKeyCols]);
        }

        // reduce frame columns to keyCols and numAggCols (without copying
        // values or the idx array) and reorder them accordingly
        Frame *reduced{};
        auto sel = DataObjectFactory::create<DenseMatrix<size_t>>(numColsRes, 1, idxs);
        extractCol(reduced, arg, sel, ctx);
        DataObjectFactory::destroy(sel);

        std::iota(idxs.get(), idxs.get() + numColsRes, 0);
        auto groups = new std::vector<std::pair<size_t, size_t>>;
        Frame *ordered{};

        // order frame rows by groups and get the group vector;
        if (numKeyCols > 0) {
            order(ordered, reduced, idxs.get(), numKeyCols, ascending, numKeyCols, false, ctx, groups);
            DataObjectFactory::destroy(reduced);
        } else {
            // skip for pure aggregation over all rows (no grouping)
            groups->push_back(std::make_pair(0, numRowsArg));
            ordered = reduced;
        }
        delete[] ascending;
        size_t inGroups = 0;
        for (auto &group : *groups) {
            inGroups += group.second - group.first;
        }
        numRowsRes -= inGroups - groups->size();

        // create the result frame
        std::string *labels = new std::string[numColsRes];
        ValueTypeCode *schema = new ValueTypeCode[numColsRes];
        if (starLabels.size()) {
            for (size_t i = 0; i < numKeyCols; i++) {
                labels[i] = starLabels[i];
                schema[i] = ordered->getColumnType(idxs[i]);
            }
        } else {
            for (size_t i = 0; i < numKeyCols; i++) {
                labels[i] = keyCols[i];
                schema[i] = ordered->getColumnType(idxs[i]);
            }
        }
        using mlir::daphne::GroupEnum;
        for (size_t i = numKeyCols; i < numColsRes; i++) {
            // TODO Maybe we can find a good way to call
            // mlir::daphne::stringifyGroupEnum, we would need to link with the
            // respective library.
            //            labels[i] =
            //            mlir::daphne::stringifyGroupEnum(aggFuncs[i-numKeyCols]).str()
            //            + "(" +  aggCols[i-numKeyCols] + ")";
            labels[i] = myStringifyGroupEnum(aggFuncs[i - numKeyCols]) + "(" + aggCols[i - numKeyCols] + ")";
            switch (aggFuncs[i - numKeyCols]) {
            case GroupEnum::COUNT:
                schema[i] = ValueTypeCode::UI64;
                break;
            case GroupEnum::SUM:
                schema[i] = ordered->getColumnType(idxs[i]);
                break;
            case GroupEnum::MIN:
                schema[i] = ordered->getColumnType(idxs[i]);
                break;
            case GroupEnum::MAX:
                schema[i] = ordered->getColumnType(idxs[i]);
                break;
            case GroupEnum::AVG:
                schema[i] = ValueTypeCode::F64;
                break;
            }
        }

        res = DataObjectFactory::create<Frame>(numRowsRes, numColsRes, schema, labels, false);
        delete[] labels;
        delete[] schema;

        // copying key columns and column-wise group aggregation
        for (size_t i = 0; i < numColsRes; i++) {
            DeduceValueTypeAndExecute<ColumnGroupAgg>::apply(
                res->getSchema()[i], ordered->getSchema()[i], res, ordered, i, groups,
                (i < numKeyCols) ? (GroupEnum)0 : aggFuncs[i - numKeyCols], ctx);
        }
        delete groups;
        DataObjectFactory::destroy(ordered);
    }
};

// Custom hash function for std::vector<std::string>
struct VectorStringHash {
    std::size_t operator()(const std::vector<std::string> &vec) const {
        std::size_t seed = 0;
        for (const auto &str : vec) {
            seed ^= std::hash<std::string>()(str) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

// Custom equality function for std::vector<std::string>
struct VectorStringEqual {
    bool operator()(const std::vector<std::string> &lhs, const std::vector<std::string> &rhs) const {
        return lhs == rhs;
    }
};

static std::vector<std::string> createGroupKeyFromArgRow(const Frame *arg, int64_t row_idx, int64_t *groupColIdxArg,
                                                         int64_t numGroupCols) {
    auto argSchema = arg->getSchema();
    std::vector<std::string> groupKey;
    for (int i = 0; i < numGroupCols; i++) {
        size_t colIdx = groupColIdxArg[i];
        if (argSchema[colIdx] == ValueTypeCode::F64) {
            float arg_value = arg->getColumn<float>(colIdx)->get(row_idx, 0);
            groupKey.push_back(castSca<std::string, float>(arg_value, nullptr));
        }
        if (argSchema[colIdx] == ValueTypeCode::SI64) {
            int64_t arg_value = arg->getColumn<int64_t>(colIdx)->get(row_idx, 0);
            groupKey.push_back(castSca<std::string, int64_t>(arg_value, nullptr));
        }
        if (argSchema[colIdx] == ValueTypeCode::STR) {
            groupKey.push_back(arg->getColumn<std::string>(colIdx)->get(row_idx, 0));
        }
    }
    return groupKey;
}

template <typename VTCol>
static void addNewValueToResult(Frame *res, const Frame *arg, int64_t row_idx_arg, int64_t row_idx_res,
                                int64_t colIdx_arg, int64_t colIdx_res) {
    auto argSchema = arg->getSchema();
    if (argSchema[colIdx_arg] == ValueTypeUtils::codeFor<VTCol>) {
        const auto arg_value = arg->getColumn<VTCol>(colIdx_arg)->get(row_idx_arg, 0);
        res->getColumn<VTCol>(colIdx_res)->set(row_idx_res, 0, arg_value);
    }
}

// Since only supporting SUM, just add two values
template <typename VTCol>
static void UpdateAggColInResult(Frame *res, const Frame *arg, int64_t row_idx_arg, int64_t row_idx_res,
                                 int64_t colIdx_arg, int64_t colIdx_res) {
    auto argSchema = arg->getSchema();
    if (argSchema[colIdx_arg] == ValueTypeUtils::codeFor<VTCol>) {
        auto result_value = res->getColumn<VTCol>(colIdx_res)->get(row_idx_res, 0);
        const auto arg_value = arg->getColumn<VTCol>(colIdx_arg)->get(row_idx_arg, 0);
        res->getColumn<VTCol>(colIdx_res)->set(row_idx_res, 0, arg_value + result_value);
    }
}

inline void groupSum(
    // results
    Frame *&res,
    // input frames
    const Frame *arg,
    // input agg column names
    const char *aggCol,
    // input group by columns
    const char **groupCols, size_t numGroupCols,
    // context
    DCTX(ctx)) {

    // Perhaps check if res already allocated.
    const size_t numRow = arg->getNumRows();

    int64_t col_idx_res = 0;
    int64_t row_idx_res = 0;

    // get columns indexes in arg
    int64_t aggColIdx = arg->getColumnIdx(aggCol);
    int64_t groupColIdxArg[numGroupCols];
    for (int64_t i = 0; i < numGroupCols; i++)
        groupColIdxArg[i] = arg->getColumnIdx(groupCols[i]);

    // Set up schema and labels for result Frame
    ValueTypeCode schema[numGroupCols + 1];
    std::string newlabels[numGroupCols + 1];

    for (int64_t col_idx_arg = 0; col_idx_arg < numGroupCols; col_idx_arg++) {
        schema[col_idx_res] = arg->getColumnType(groupColIdxArg[col_idx_arg]);
        newlabels[col_idx_res++] = groupCols[col_idx_arg];
    }
    schema[numGroupCols] = arg->getColumnType(aggColIdx);
    newlabels[numGroupCols] = std::string("SUM(") + std::string(aggCol) + std::string(")");

    // Initialize result frame
    res = DataObjectFactory::create<Frame>(numRow, numGroupCols + 1, schema, newlabels, false);

    // unordered_map to store group colomns and its related row index in arg Frame.
    std::unordered_map<std::vector<std::string>, int64_t, VectorStringHash, VectorStringEqual> colMap;

    for (int64_t row_idx_arg = 0; row_idx_arg < numRow; row_idx_arg++) {
        auto groupKey = createGroupKeyFromArgRow(arg, row_idx_arg, groupColIdxArg, numGroupCols);
        if (colMap.find(groupKey) == colMap.end()) {
            // add a new row to result for new columns group
            colMap[groupKey] = row_idx_res;
            for (int64_t Colidx = 0; Colidx < numGroupCols; Colidx++) {
                addNewValueToResult<std::string>(res, arg, row_idx_arg, row_idx_res, groupColIdxArg[Colidx], Colidx);
                addNewValueToResult<float>(res, arg, row_idx_arg, row_idx_res, groupColIdxArg[Colidx], Colidx);
                addNewValueToResult<int64_t>(res, arg, row_idx_arg, row_idx_res, groupColIdxArg[Colidx], Colidx);
            }
            addNewValueToResult<float>(res, arg, row_idx_arg, row_idx_res, aggColIdx, numGroupCols);
            addNewValueToResult<int64_t>(res, arg, row_idx_arg, row_idx_res, aggColIdx, numGroupCols);
            row_idx_res++;
        } else {
            auto res_row_indx = colMap[groupKey];
            UpdateAggColInResult<int64_t>(res, arg, row_idx_arg, res_row_indx, aggColIdx, numGroupCols);
            UpdateAggColInResult<float>(res, arg, row_idx_arg, res_row_indx, aggColIdx, numGroupCols);
        }
    }

    // Shrink result frame to actual size
    res->shrinkNumRows(row_idx_res);
}

#endif // SRC_RUNTIME_LOCAL_KERNELS_GROUP_H
