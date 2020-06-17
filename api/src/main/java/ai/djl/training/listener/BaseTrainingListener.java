package ai.djl.training.listener;

import ai.djl.training.Trainer;

/**
 * Base implementation of the training listener that does nothing. This is to be used as a base
 * class for custom training listeners that just want to listen to one event, so it is not necessary
 * to override methods you do not care for.
 */
public abstract class BaseTrainingListener implements TrainingListener {
    @Override
    public void onEpoch(Trainer trainer) {}

    @Override
    public void onTrainingBatch(Trainer trainer, BatchData batchData) {}

    @Override
    public void onValidationBatch(Trainer trainer, BatchData batchData) {}

    @Override
    public void onTrainingBegin(Trainer trainer) {}

    @Override
    public void onTrainingEnd(Trainer trainer) {}
}
