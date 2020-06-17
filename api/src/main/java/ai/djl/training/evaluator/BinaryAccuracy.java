/*
 * Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */
package ai.djl.training.evaluator;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.util.Pair;

/**
 * {@link BinaryAccuracy} is the {@link AbstractAccuracy} with two classes.
 *
 * <p>It is assumed that the classes are identified with a labels array of 0s and 1s and a
 * prediction array where values above the threshold are the positive (1) examples and values below
 * the threshold are the negative (0) examples. If you have a different encoding, you may want to
 * look at the {@link Accuracy}.
 */
public class BinaryAccuracy extends AbstractAccuracy {

    float threshold;

    /**
     * Creates a binary (two class) accuracy evaluator.
     *
     * @param manager an {@link NDManager}
     * @param name the name of the evaluator, default is "Accuracy"
     * @param threshold the value differentiating the posive and negative classes (usually 0 or .5)
     * @param index the index of the NDArray in labels to compute accuracy for
     * @param axis the axis that represent classes in prediction, default 1
     */
    public BinaryAccuracy(NDManager manager, String name, float threshold, int index, int axis) {
        super(manager, name, index, axis);
        this.threshold = threshold;
    }

    /**
     * Creates a binary (two class) accuracy evaluator that computes accuracy across axis 1 along
     * given index.
     *
     * @param manager an {@link NDManager}
     * @param name the name of the evaluator, default is "Accuracy"
     * @param threshold the value differentiating the posive and negative classes (usually 0 or .5)
     * @param index the index of the NDArray in labels to compute accuracy for
     */
    public BinaryAccuracy(NDManager manager, String name, float threshold, int index) {
        this(manager, name, threshold, index, 1);
    }

    /**
     * Creates a binary (two class) accuracy evaluator that computes accuracy across axis 1 along
     * the 0th index.
     *
     * @param manager an {@link NDManager}
     * @param threshold the value differentiating the posive and negative classes (usually 0 or .5)
     */
    public BinaryAccuracy(NDManager manager, float threshold) {
        this(manager, "BinaryAccuracy", threshold, 0, 1);
    }

    /**
     * Creates a binary (two class) accuracy evaluator with 0 threshold.
     *
     * @param manager an {@link NDManager}
     */
    public BinaryAccuracy(NDManager manager) {
        this(manager, 0);
    }

    /** {@inheritDoc} */
    @Override
    protected Pair<Long, NDArray> accuracyHelper(NDList labels, NDList predictions) {
        if (labels.size() != predictions.size()) {
            throw new IllegalArgumentException("labels and prediction length does not match.");
        }
        NDArray label = labels.get(index);
        NDArray prediction = predictions.get(index);
        checkLabelShapes(label, prediction, false);
        NDArray predictionReduced = prediction.gte(threshold);
        // result of sum is int64 now
        long total = label.size();
        NDArray correct =
                label.toType(DataType.INT64, false)
                        .eq(predictionReduced.toType(DataType.INT64, false))
                        .countNonzero();
        return new Pair<>(total, correct);
    }
}
