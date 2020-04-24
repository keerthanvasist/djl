package ai.djl.modality.nlp.translator;

import ai.djl.Model;
import ai.djl.modality.nlp.Decoder;
import ai.djl.modality.nlp.Encoder;
import ai.djl.modality.nlp.EncoderDecoder;
import ai.djl.modality.nlp.embedding.EmbeddingException;
import ai.djl.modality.nlp.embedding.TrainableTextEmbedding;
import ai.djl.modality.nlp.preprocess.LowerCaseConvertor;
import ai.djl.modality.nlp.preprocess.PunctuationSeparator;
import ai.djl.modality.nlp.preprocess.SentenceLengthNormalizer;
import ai.djl.modality.nlp.preprocess.SimpleTokenizer;
import ai.djl.modality.nlp.preprocess.TextProcessor;
import ai.djl.modality.nlp.preprocess.TextTruncator;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Block;
import ai.djl.nn.BlockList;
import ai.djl.nn.SequentialBlock;
import ai.djl.translate.Batchifier;
import ai.djl.translate.PaddingStackBatchifier;
import ai.djl.translate.PreProcessor;
import ai.djl.translate.TranslateException;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorContext;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Locale;

public class Seq2SeqTranslator implements Translator<String, String> {
    private SimpleTokenizer tokenizer = new SimpleTokenizer();
    private TrainableTextEmbedding sourceEmbedding;
    private TrainableTextEmbedding targetEmbedding;
    private List<TextProcessor> textProcessors = Arrays.asList(
            new SimpleTokenizer(),
            new LowerCaseConvertor(Locale.ENGLISH),
            new PunctuationSeparator(),
            new SentenceLengthNormalizer());


    /** {@inheritDoc} */
    @Override
    public String processOutput(TranslatorContext ctx, NDList list) {
        if (list.singletonOrThrow().getShape().dimension() > 2) {
            throw new IllegalArgumentException("Input must correspond to one sentence. Shape must be of 2 or less dimensions");
        }
        if (targetEmbedding == null) {
            Model model = ctx.getModel();
            EncoderDecoder encoderDecoder = (EncoderDecoder) model.getBlock();
            BlockList children = encoderDecoder.getChildren();
            Decoder decoder = (Decoder) children.get(1).getValue();
            SequentialBlock sequentialBlock = (SequentialBlock) decoder.getChildren().get(0).getValue();
            targetEmbedding = (TrainableTextEmbedding) sequentialBlock.getChildren().get(0).getValue();
        }
        List<String> output = new ArrayList<>();
        for (String token : targetEmbedding.unembedText(list.singletonOrThrow().toType(DataType.INT32, false).flatten())) {
            if (token.equals("<eos>")) {
                break;
            }
            output.add(token);
        }
        return tokenizer.buildSentence(output);
    }

    /** {@inheritDoc} */
    @Override
    public NDList processInput(TranslatorContext ctx, String input) {
        Model model = ctx.getModel();
        if (sourceEmbedding == null) {
            EncoderDecoder encoderDecoder = (EncoderDecoder) model.getBlock();
            BlockList children = encoderDecoder.getChildren();
            Encoder encoder = (Encoder) children.get(0).getValue();
            SequentialBlock sequentialBlock = (SequentialBlock) encoder.getChildren().get(0).getValue();
            sourceEmbedding = (TrainableTextEmbedding) sequentialBlock.getChildren().get(0).getValue();
        }
        List<String> tokens = Collections.singletonList(input);
        for (TextProcessor textProcessor : textProcessors) {
            tokens = textProcessor.preprocess(tokens);
        }
        return new NDList(model.getNDManager().create(sourceEmbedding.preprocessTextToEmbed(tokens)), model.getNDManager().create(sourceEmbedding.preprocessTextToEmbed(Arrays.asList("<bos>"))));
    }

    /** {@inheritDoc} */
    @Override
    public Batchifier getBatchifier() {
        return PaddingStackBatchifier.builder()
                .optIncludeValidLengths(false)
                .addPad(0, 0, (m) -> m.ones(new Shape(1)).mul(sourceEmbedding.preprocessTextToEmbed(Arrays.asList("<pad>"))[0]), 10)
                .build();
    }
}
