import { Card } from "@/components/ui";

export function MlNextSteps() {
  return <Card title="From readiness to a tested model">
    <p>Passing readiness makes the data eligible for preparation. It does not start training or change AI Confidence.</p>
    <ol>
      <li><strong>Finalize a fresh dataset.</strong> Review and save the exact examples to use. An older saved dataset keeps its original gaps even when today’s live readiness passes.</li>
      <li><strong>Run preparation checks.</strong> Select that dataset and check its actual training target, class balance and reserved final test.</li>
      <li><strong>Train the challengers.</strong> Train XGBoost, LightGBM and CatBoost on the same examples and retain their artifacts.</li>
      <li><strong>Evaluate.</strong> Compare on unseen periods and against the saved production confidence. Review development results before explicitly requesting the reserved final test.</li>
      <li><strong>Shadow-test.</strong> Register reviewed candidates and score new signals without changing the app’s production model.</li>
      <li><strong>Review promotion.</strong> Passing shadow checks permits a promotion decision. Connecting a model to production AI Confidence is a separate change.</li>
    </ol>
    <p className="cap"><a href="https://github.com/realhowieb/ai_scanner/actions/workflows/ml-v4-train.yml" target="_blank" rel="noopener noreferrer">Open manual preparation, training and evaluation workflow</a>. Nothing runs automatically from this checklist.</p>
  </Card>;
}
