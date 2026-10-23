# Evaluation Plan: Meeting Sentiment & Emotion Accuracy

## 1. Objective
Validate the accuracy of the `GeminiIntelligenceService` emotion and sentiment extraction against human-labeled ground truth. Ensure that the "evidence-based" approach reduces fabrication and increases reliability for research-grade results.

## 2. Dataset Selection
- **Sample Size**: 20-50 representative meeting recordings (varying length and speaker count).
- **Ground Truth**: Human annotators (at least 2) will label the following for a random 10% of segments in each meeting:
    - Dominant Emotion (from the taxonomy: neutral, happy, excited, sad, angry, frustrated, concerned, nervous, anxiety, confusion, urgency, calmness, professional).
    - Overall Meeting Sentiment (positive, neutral, concerned, negative).
- **Inter-Annotator Agreement**: Measured using Cohen's Kappa.

## 3. Evaluation Metrics
We will treat emotion detection as a multi-class classification problem.

### 3.1 Quantitative Metrics
- **Per-Class F1-Score**: Precision, Recall, and F1 for each emotion in the taxonomy.
- **Weighted Macro-F1**: To account for class imbalance (e.g., 'professional' is likely more common than 'angry').
- **Sentiment Accuracy**: Percentage of meetings where the `overall_sentiment` matches the human label.
- **Evidence Validity Rate**: Percentage of LLM-provided quotes that are verbatim substrings of the transcript (this should be 100% given the new verification logic).

### 3.2 Qualitative Analysis
- **Error Matrix**: A confusion matrix to see which emotions are most frequently confused (e.g., 'concerned' vs 'frustrated').
- **Fabrication Rate**: Frequency of "hallucinated" evidence before vs after implementing the `_verify_evidence` guard.

## 4. Baseline Comparison
Compare Gemini's performance against:
1. **Zero-Shot Prompting**: Without the evidence requirement.
2. **Traditional Sentiment Analysis**: VADER or a pretrained BERT-based emotion classifier (e.g., `bhadresh-savani/distilbert-base-uncased-emotion`).

## 5. Procedure
1. **Collection**: Process the selected meetings through the MeetWise pipeline.
2. **Annotation**: Human annotators label the ground truth.
3. **Extraction**: Run `GeminiIntelligenceService.analyze` to get predictions.
4. **Matching**: Align LLM labels with human labels.
5. **Calculation**: Compute metrics and generate the report.

## 6. Success Criteria
- **Emotion F1-Score**: $> 0.70$ for dominant categories.
- **Evidence Validity**: $100\%$ (Zero fabrication).
- **Sentiment Accuracy**: $> 80\%$ agreement with human labels.
