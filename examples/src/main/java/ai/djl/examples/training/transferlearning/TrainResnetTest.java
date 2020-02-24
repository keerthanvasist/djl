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
package ai.djl.examples.training.transferlearning;

import ai.djl.MalformedModelException;
import ai.djl.examples.training.util.ExampleTrainingResult;
import ai.djl.repository.zoo.ModelNotFoundException;
import java.io.IOException;
import org.apache.commons.cli.ParseException;

public final class TrainResnetTest {
    private TrainResnetTest() {}

    public static void main(String[] args)
            throws IOException, ParseException, ModelNotFoundException, MalformedModelException {
        args = new String[] {"-e", "10", "-g", "4"};

        ExampleTrainingResult result = TrainResnetWithCifar10.runExample(args);
        System.out.println(result.getEvaluation("Accuracy")); // NOPMD
        System.out.println(result.getEvaluation("SoftmaxCrossEntropyLoss")); // NOPMD
    }
}
