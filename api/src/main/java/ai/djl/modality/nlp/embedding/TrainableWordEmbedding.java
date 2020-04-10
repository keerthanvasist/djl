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
package ai.djl.modality.nlp.embedding;

import ai.djl.MalformedModelException;
import ai.djl.modality.nlp.SimpleVocabulary;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.BlockList;
import ai.djl.nn.Parameter;
import ai.djl.nn.core.Embedding;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.Collections;
import java.util.List;

/**
 * {@code VocabWordEmbedding} is an implementation of {@link WordEmbedding} based on a Vocabulary or
 * an {@link Embedding} block. This {@link WordEmbedding} is ideal when there is no pre-trained
 * embeddings available, or when the pre-trained embedding needs to further trained.
 */
public class TrainableWordEmbedding extends AbstractBlock implements WordEmbedding {
    private static final String DEFAULT_UNKNOWN_TOKEN = "<unk>";

    private Embedding<String> embedding;
    private String unknownToken;

    /**
     * Constructs a new instance {@code VocabWordEmbedding} from a given {@link Embedding} block.
     *
     * @param embedding the {@link Embedding} block
     */
    public TrainableWordEmbedding(Embedding<String> embedding) {
        this(embedding, DEFAULT_UNKNOWN_TOKEN);
    }

    /**
     * Constructs a new instance {@code VocabWordEmbedding} from a given {@link Embedding} block.
     *
     * @param embedding the {@link Embedding} block
     * @param unknownToken the {@link String} value of unknown token
     */
    public TrainableWordEmbedding(Embedding<String> embedding, String unknownToken) {
        this.embedding = embedding;
        this.unknownToken = unknownToken;
    }

    /**
     * Constructs a new instance {@code VocabWordEmbedding} based on the given {@link
     * SimpleVocabulary} and embedding size.
     *
     * @param vocabulary the {@link SimpleVocabulary} based on which the embedding is built.
     * @param embeddingSize the size of the embedding for each word
     */
    public TrainableWordEmbedding(SimpleVocabulary vocabulary, int embeddingSize) {
        this(vocabulary.newEmbedding(embeddingSize), DEFAULT_UNKNOWN_TOKEN);
    }

    /** {@inheritDoc} */
    @Override
    public boolean vocabularyContains(String word) {
        return embedding.hasItem(word);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray preprocessWordToEmbed(NDManager manager, String word) {
        if (embedding.hasItem(word)) {
            return embedding.embed(manager, word);
        }
        return embedding.embed(manager, unknownToken);
    }

    /** {@inheritDoc} */
    @Override
    public NDArray embedWord(NDArray word) {
        throw new UnsupportedOperationException("This operation is not supported by this class.");
    }

    /** {@inheritDoc} */
    @Override
    public String unembedWord(NDArray wordEmbedding) {
        throw new UnsupportedOperationException("This operation is not supported yet.");
    }

    /** {@inheritDoc} */
    @Override
    public NDList forward(
            ParameterStore parameterStore, NDList inputs, PairList<String, Object> params) {
        return embedding.forward(parameterStore, inputs, params);
    }

    /** {@inheritDoc} */
    @Override
    public Shape[] initialize(NDManager manager, DataType dataType, Shape... inputShapes) {
        return embedding.initialize(manager, dataType, inputShapes);
    }

    /** {@inheritDoc} */
    @Override
    public BlockList getChildren() {
        return new BlockList(
                Collections.singletonList("embedding"), Collections.singletonList(embedding));
    }

    /** {@inheritDoc} */
    @Override
    public List<Parameter> getDirectParameters() {
        return Collections.emptyList();
    }

    /** {@inheritDoc} */
    @Override
    public Shape getParameterShape(String name, Shape[] inputShapes) {
        throw new IllegalArgumentException("TrainableWordEmbeddings have no parameters");
    }

    /** {@inheritDoc} */
    @Override
    public Shape[] getOutputShapes(NDManager manager, Shape[] inputShapes) {
        return embedding.getOutputShapes(manager, inputShapes);
    }

    /** {@inheritDoc} */
    @Override
    public void saveParameters(DataOutputStream os) throws IOException {
        embedding.saveParameters(os);
    }

    /** {@inheritDoc} */
    @Override
    public void loadParameters(NDManager manager, DataInputStream is)
            throws IOException, MalformedModelException {
        embedding.loadParameters(manager, is);
    }
}
