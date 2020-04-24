/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
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
package ai.djl.examples.training.util;

import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.Block;
import ai.djl.nn.Parameter;
import ai.djl.nn.ParameterList;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.Dataset;
import ai.djl.util.Pair;
import java.io.IOException;
import java.nio.file.Paths;

public final class TrainingUtils {

    private TrainingUtils() {}

    public static void fit(
            Trainer trainer,
            int numEpoch,
            Dataset trainingDataset,
            Dataset validateDataset,
            String outputDir,
            String modelName)
            throws IOException {
        for (int epoch = 0; epoch < numEpoch; epoch++) {
            for (Batch batch : trainer.iterateDataset(trainingDataset)) {
                trainer.trainBatch(batch);
                trainer.step();
                batch.close();
            }

            if (validateDataset != null) {
                for (Batch batch : trainer.iterateDataset(validateDataset)) {
                    trainer.validateBatch(batch);
                    batch.close();
                }
            }
            // reset training and validation evaluators at end of epoch
            trainer.endEpoch();
            // save model at end of each epoch
            if (outputDir != null) {
                Model model = trainer.getModel();
                model.setProperty("Epoch", String.valueOf(epoch));
                model.save(Paths.get(outputDir), modelName);
            }
        }
    }

    public static void gradientClipping(NDManager manager, Block block, double theta) {
        manager = manager.newSubManager();
        ParameterList parameters = block.getParameters();
        NDArray norm = manager.create(0);
        for (Pair<String, Parameter> parameter : parameters) {
            NDArray gradient = parameter.getValue().getArray().getGradient();
            norm = norm.toType(gradient.getDataType(), false);
            norm = norm.add(gradient.square().sum());
        }
        norm = norm.sqrt();
        float normVal = norm.getFloat();
        if (normVal > theta) {
            for (Pair<String, Parameter> parameter : parameters) {
                parameter.getValue().getArray().getGradient().muli(theta / normVal);
            }
        }
        manager.close();
    }
}
