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

package ai.djl.examples.training;

import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.examples.training.transferlearning.TrainResnetWithCifar10;
import ai.djl.examples.training.util.ExampleTrainingResult;
import ai.djl.repository.zoo.ModelNotFoundException;
import java.io.IOException;
import org.apache.commons.cli.ParseException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.SkipException;
import org.testng.annotations.Test;

public class TrainResNetTest {

    @Test
    public void testTrainResNet()
            throws ParseException, ModelNotFoundException, IOException, MalformedModelException {
        // Limit max 4 gpu for cifar10 training to make it converge faster.
        // and only train 10 batch for unit test.
        String[] args = {"-e", "2", "-g", "4", "-m", "10", "-s", "-p"};

        TrainResnetWithCifar10.runExample(args);
    }

    @Test
    public void testTrainResNetSymbolicNightly()
            throws ParseException, ModelNotFoundException, IOException, MalformedModelException {
        // this is nightly test
        if (!Boolean.getBoolean("nightly")) {
            throw new SkipException("Nightly only");
        }
        if (Device.getGpuCount() > 0) {
            // Limit max 4 gpu for cifar10 training to make it converge faster.
            // and only train 10 batch for unit test.
            String[] args = {"-e", "10", "-g", "4", "-s", "-p"};

            ExampleTrainingResult result = TrainResnetWithCifar10.runExample(args);
            Logger logger = LoggerFactory.getLogger(TrainResNetTest.class);
            logger.info("Accuracy={}", result.getEvaluation("Accuracy"));
            logger.info(
                    "SoftmaxCrossEntropyLoss={}", result.getEvaluation("SoftmaxCrossEntropyLoss"));
            Assert.assertTrue(result.getEvaluation("Accuracy") > 0.67f);
            Assert.assertTrue(result.getEvaluation("SoftmaxCrossEntropyLoss") < 1.1);
        }
    }

    @Test
    public void testTrainResNetImperativeNightly()
            throws ParseException, ModelNotFoundException, IOException, MalformedModelException {
        // this is nightly test
        if (!Boolean.getBoolean("nightly")) {
            throw new SkipException("Nightly only");
        }
        if (Device.getGpuCount() > 0) {
            // Limit max 4 gpu for cifar10 training to make it converge faster.
            // and only train 10 batch for unit test.
            String[] args = {"-e", "10", "-g", "4"};

            ExampleTrainingResult result = TrainResnetWithCifar10.runExample(args);
            Assert.assertTrue(result.getEvaluation("Accuracy") > 0.7f);
            Assert.assertTrue(result.getEvaluation("SoftmaxCrossEntropyLoss") < 0.9);
        }
    }
}
