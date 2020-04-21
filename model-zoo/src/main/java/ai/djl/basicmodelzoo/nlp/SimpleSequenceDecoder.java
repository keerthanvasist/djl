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
package ai.djl.basicmodelzoo.nlp;

import ai.djl.modality.nlp.Decoder;
import ai.djl.modality.nlp.embedding.TrainableTextEmbedding;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.nn.recurrent.RecurrentBlock;
import ai.djl.training.ParameterStore;

/**
 * {@code SimpleSequenceDecoder} implements a {@link Decoder} that employs a {@link RecurrentBlock}
 * to decode text input.
 */
public class SimpleSequenceDecoder extends Decoder {
    private RecurrentBlock recurrentBlock;
    private TrainableTextEmbedding trainableTextEmbedding;
    private String eosToken;
    private String bosToken;

    /**
     * Contructs a new instance of {@code SimpleSequenceDecoder} with the given {@link
     * RecurrentBlock}. Use this constructor if you are planning to use pre-trained embeddings that
     * don't need further training.
     *
     * @param recurrentBlock the recurrent block to be used to decode
     * @param vocabSize the size of the {@link ai.djl.modality.nlp.Vocabulary}
     */
    public SimpleSequenceDecoder(
            RecurrentBlock recurrentBlock, int vocabSize, String eosToken, String bosToken) {
        this(null, recurrentBlock, vocabSize, eosToken, bosToken);
    }

    /**
     * Contructs a new instance of {@code SimpleSequenceDecoder} with the given {@link
     * RecurrentBlock}. Use this constructor if you are planning to use pre-trained embeddings that
     * don't need further training.
     *
     * @param trainableTextEmbedding the {@link TrainableTextEmbedding} to train embeddings with
     * @param recurrentBlock the recurrent block to be used to decode
     * @param vocabSize the size of the {@link ai.djl.modality.nlp.Vocabulary}
     */
    public SimpleSequenceDecoder(
            TrainableTextEmbedding trainableTextEmbedding,
            RecurrentBlock recurrentBlock,
            int vocabSize,
            String eosToken,
            String bosToken) {
        super(getBlock(trainableTextEmbedding, recurrentBlock, vocabSize));
        this.recurrentBlock = recurrentBlock;
        this.trainableTextEmbedding = trainableTextEmbedding;
        this.eosToken = eosToken;
        this.bosToken = bosToken;
    }

    private static Block getBlock(
            TrainableTextEmbedding trainableTextEmbedding,
            RecurrentBlock recurrentBlock,
            int vocabSize) {
        SequentialBlock sequentialBlock = new SequentialBlock();
        sequentialBlock
                .add(trainableTextEmbedding)
                .add(recurrentBlock)
                .add(Linear.builder().setOutChannels(vocabSize).optFlatten(false).build());
        return sequentialBlock;
    }

    /** {@inheritDoc} */
    @Override
    public void initState(NDArray encoderState) {
        recurrentBlock.setBeginState(encoderState);
    }

    /** {@inheritDoc} */
    @Override
    public NDList predict(ParameterStore parameterStore, NDList inputs) {
        Shape inputShape = inputs.get(0).getShape();
        if (inputShape.get(0) != 1 && inputShape.get(1) != 1) {
            throw new IllegalArgumentException(
                    "Input to predict must be of shape (1, 1, <embeddingSize>");
        }
        String token = eosToken;
        NDList output = new NDList();
        int i = 0;
        while (!token.equals(bosToken) && i < 10) {
            NDList outputList = forward(parameterStore, inputs);
            // System.out.println(outputList.head().getShape());
            // System.out.println("Input to TextEmbedding" +
            // outputList.head().argMax(2).getShape());
            token =
                    trainableTextEmbedding
                            .unembedText(outputList.head().argMax(2).toType(DataType.INT32, false))
                            .get(0);
            System.out.print(token + " ");
            inputs = new NDList(outputList.head().argMax(2));
            output.add(outputList.singletonOrThrow().argMax(2));
            i++;
        }
        System.out.println();
        return new NDList(NDArrays.concat(output, 1));
    }
}
