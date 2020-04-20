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
import ai.djl.training.Trainer;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.Dataset;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

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
        Logger logger = LoggerFactory.getLogger(TrainingUtils.class);
        long averageLoad = 0;
        long averageBatch = 0;
        long averageSplit = 0;
        long time = System.nanoTime();
        int i = 1;
        for (int epoch = 0; epoch < numEpoch; epoch++) {
            for (Batch batch : trainer.iterateDataset(trainingDataset)) {
                averageLoad = (averageLoad * (i-1) + System.nanoTime() - time)/i;
                logger.info("Train load={}", averageLoad);
                time = System.nanoTime();
                trainer.trainBatch(batch);
                averageBatch = (averageBatch * (i-1) + System.nanoTime() - time)/i;
                logger.info("Train batch={}", averageBatch);
                time = System.nanoTime();
                trainer.step();
                averageSplit = (averageSplit * (i-1) + System.nanoTime() - time)/i;
                logger.info("Train step={}", averageSplit);
                i++;
                time = System.nanoTime();
                //logger.info("\nBatch time={}", trainer.getMetrics().getMetric("train"));
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
}
