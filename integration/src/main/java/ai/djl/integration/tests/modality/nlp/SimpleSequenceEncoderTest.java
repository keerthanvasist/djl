package ai.djl.integration.tests.modality.nlp;

import ai.djl.basicmodelzoo.nlp.SimpleSequenceEncoder;
import ai.djl.modality.nlp.embedding.TrainableTextEmbedding;
import ai.djl.modality.nlp.embedding.TrainableWordEmbedding;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.recurrent.LSTM;
import ai.djl.training.ParameterStore;
import ai.djl.training.initializer.XavierInitializer;
import java.util.Arrays;
import org.testng.Assert;
import org.testng.annotations.Test;

public class SimpleSequenceEncoderTest {
    @Test
    public void testEncoder() {
        TrainableTextEmbedding trainableTextEmbedding =
                new TrainableTextEmbedding(
                        TrainableWordEmbedding.builder()
                                .setEmbeddingSize(8)
                                .setItems(Arrays.asList("1 2 3 4 5 6 7 8 9 10".split(" ")))
                                .build());
        SimpleSequenceEncoder encoder =
                new SimpleSequenceEncoder(
                        trainableTextEmbedding,
                        LSTM.builder()
                                .setNumStackedLayers(2)
                                .setSequenceLength(false)
                                .setStateSize(16)
                                .build());
        try (NDManager manager = NDManager.newBaseManager()) {
            encoder.setInitializer(new XavierInitializer());
            encoder.initialize(manager, DataType.FLOAT32, new Shape(4, 7));
            NDList output =
                    encoder.forward(
                            new ParameterStore(manager, false),
                            new NDList(manager.zeros(new Shape(4, 7))));
            Assert.assertEquals(output.head().getShape(), new Shape(4, 7, 16));
            Assert.assertEquals(output.size(), 3);
            Assert.assertEquals(output.get(1).getShape(), new Shape(2, 4, 16));
            Assert.assertEquals(output.get(2).getShape(), new Shape(2, 4, 16));
        }
    }
}
