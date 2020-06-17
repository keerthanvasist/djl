package ai.djl.training.optimizer.learningrate;

public class PolynomialDecayLearningRateTracker extends LearningRateTracker {

    protected float endLearningRate;
    protected int decaySteps;
    protected float power;

    public PolynomialDecayLearningRateTracker(final Builder builder) {
        super(builder);
        if (Float.isNaN(builder.endLearningRate)) {
            throw new IllegalArgumentException("End learning rate is not set.");
        }
        if (builder.decaySteps <= 0) {
            throw new IllegalArgumentException("Decay steps is not set.");
        }
        this.endLearningRate = builder.endLearningRate;
        this.decaySteps = builder.decaySteps;
        this.power = builder.power;
    }


    @Override
    public float getNewLearningRate(final int numUpdate) {
        if (numUpdate < warmUpSteps) {
            return getWarmUpLearningRate(numUpdate);
        }
        int step = Math.max(0, Math.min(numUpdate - warmUpSteps, decaySteps));
        double decayedLearningRate = (baseLearningRate - endLearningRate) *
                Math.pow (1.0 - (double)step / (double)decaySteps, power) +
                endLearningRate;
        return (float)decayedLearningRate;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static class Builder extends LearningRateTracker.LrBaseBuilder<Builder> {

        protected float endLearningRate = Float.NaN;
        protected int decaySteps = -1;
        protected float power = 1f;

        public Builder setEndLearningRate(float endLearningRate) {
            this.endLearningRate = endLearningRate;
            return self();
        }

        public Builder setDecaySteps(int decaySteps) {
            this.decaySteps = decaySteps;
            return self();
        }

        public Builder optPower(float power) {
            this.power = power;
            return self();
        }

        public PolynomialDecayLearningRateTracker build() {
            return new PolynomialDecayLearningRateTracker(this);
        }

        @Override
        protected Builder self() {
            return this;
        }
    }
}
