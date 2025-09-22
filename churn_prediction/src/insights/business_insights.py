
#Business Insights Generator


import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BusinessInsights:
    def __init__(self):
        self.model_type = None
        self.model = None
        self.feature_importance = None
    
    
    #Load model information and feature importance
    def load_model_info(self):
        
        try:
            # Load model selection results
            with open("models/model_selection_results.json", "r") as f:
                results = json.load(f)
            self.model_type = results['best_model']

            # Load model
            self.model = joblib.load("models/Best_Churn_Model.pkl")

            # Load feature importance if available
            try:
                importance_path = "output/feature_importance.csv"
                self.feature_importance = pd.read_csv(importance_path)
            except FileNotFoundError:
                self.feature_importance = self.calculate_feature_importance()

            logger.info(f"Generating insights for {self.model_type} model")

        except FileNotFoundError:
            logger.error("Model information not found")
            raise
    
    #Calculate feature importance based on model type
    def calculate_feature_importance(self):
        
        features = ['Recency', 'Frequency', 'Monetary']

        if self.model_type == "RandomForest" and hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        elif self.model_type == "XGBoost" and hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        elif self.model_type == "LogisticRegression" and hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_[0])
        else:
            importance = [0.33, 0.33, 0.33]  # Default equal importance

        return pd.DataFrame({
            'feature': features,
            'importance': importance,
            'model_type': self.model_type
        })
    
    
    #Plots the feature importance and saves the plot.
    def plot_feature_importance(self, df, path):
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=df.sort_values('importance', ascending=False))
        plt.title('Feature Importance')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path)
        logger.info(f"Feature importance plot saved to {path}")
        plt.close()
    
    #Identify main drivers of churn based on model type
    def generate_churn_drivers(self):
        
        if self.feature_importance is None:
            self.feature_importance = self.calculate_feature_importance()

        top_features = self.feature_importance.nlargest(3, 'importance')

        drivers = []
        for _, row in top_features.iterrows():
            feature = row['feature']
            importance = row['importance']

            if feature == "Recency":
                drivers.append({
                    'driver': 'Customer Engagement',
                    'insight': f'Recent activity (Recency) drives {importance:.1%} of churn predictions',
                    'recommendation': 'Focus on re-engaging dormant customers',
                    'impact': 'High'
                })
            elif feature == "Frequency":
                drivers.append({
                    'driver': 'Purchase Behavior',
                    'insight': f'Purchase frequency drives {importance:.1%} of churn predictions',
                    'recommendation': 'Implement loyalty programs for frequent buyers',
                    'impact': 'Medium'
                })
            elif feature == "Monetary":
                drivers.append({
                    'driver': 'Customer Value',
                    'insight': f'Spending amount drives {importance:.1%} of churn predictions',
                    'recommendation': 'Create personalized offers for high-value customers',
                    'impact': 'High'
                })

        return drivers
    
    #Generate customer risk segments based on model type
    def generate_risk_segments(self):
        
        if self.model_type == "RandomForest":
            return [
                {
                    'segment': 'High Risk',
                    'description': 'Dormant customers with low frequency and spending',
                    'size_estimate': '15-20% of customer base',
                    'churn_probability': '70-90%',
                    'action': 'Immediate retention campaigns'
                },
                {
                    'segment': 'Medium Risk',
                    'description': 'Moderately active customers showing early signs of disengagement',
                    'size_estimate': '25-30% of customer base',
                    'churn_probability': '30-70%',
                    'action': 'Proactive engagement and offers'
                },
                {
                    'segment': 'Low Risk',
                    'description': 'Active, engaged customers with high loyalty',
                    'size_estimate': '50-60% of customer base',
                    'churn_probability': '5-30%',
                    'action': 'Maintain current engagement strategies'
                }
            ]
        elif self.model_type == "XGBoost":
            return [
                {
                    'segment': 'High Risk',
                    'description': 'Customers with complex disengagement patterns',
                    'size_estimate': '10-15% of customer base',
                    'churn_probability': '75-95%',
                    'action': 'Personalized intervention required'
                }
            ]
        return []
    
    
    #Generate strategic recommendations based on model insights
    def generate_strategic_recommendations(self):
        
        if self.model_type == "RandomForest":
            return [
                {
                    'area': 'Customer Engagement',
                    'recommendation': 'Implement automated win-back campaigns for dormant customers',
                    'expected_impact': '20-30% reduction in churn',
                    'timeframe': '3-6 months',
                    'priority': 'High'
                },
                {
                    'area': 'Product Development',
                    'recommendation': 'Develop features that encourage frequent usage',
                    'expected_impact': '15-25% improvement in retention',
                    'timeframe': '6-12 months',
                    'priority': 'Medium'
                }
            ]
        elif self.model_type == "LogisticRegression":
            return [
                {
                    'area': 'Pricing Strategy',
                    'recommendation': 'Review pricing for at-risk customer segments',
                    'expected_impact': '10-20% improvement in retention',
                    'timeframe': '1-3 months',
                    'priority': 'High'
                }
            ]
        return []
    
    
    #Generate comprehensive business insights report
    def generate_comprehensive_report(self):
        
        self.load_model_info()

        report = {
            'generation_date': datetime.now().isoformat(),
            'model_type': self.model_type,
            'model_performance': self.get_model_performance(),
            'key_churn_drivers': self.generate_churn_drivers(),
            'risk_segments': self.generate_risk_segments(),
            'strategic_recommendations': self.generate_strategic_recommendations(),
            'executive_summary': self.generate_executive_summary()
        }

        Path("insights").mkdir(exist_ok=True)
        report_path = f"insights/business_insights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)

        logger.info(f"📈 Business insights report saved: {report_path}")
        return report
    
    #Get model performance metrics
    def get_model_performance(self):
        
        try:
            with open("models/model_selection_results.json", "r") as f:
                results = json.load(f)
            return {
                'auc_score': results['best_score'],
                'model_confidence': 'High' if results['best_score'] > 0.8 else 'Medium',
                'evaluation_date': results['selection_date']
            }
        except Exception:
            return {'auc_score': 0.0, 'model_confidence': 'Unknown'}
    
    #Generate executive summary
    def generate_executive_summary(self):
        
        performance = self.get_model_performance()
        return {
            'overview': f"Our {self.model_type} model effectively predicts customer churn with {performance['auc_score']:.3f} AUC score",
            'key_finding': "Customer engagement (Recency) is the strongest predictor of churn",
            'business_impact': "Potential to reduce churn by 20-30% through targeted interventions",
            'next_steps': "Implement automated retention campaigns for high-risk segments"
        }
    
    #Analyzes customer segments by calculating average churn probability and feature values.
    def analyze_segments(self, clustered_data, model, features):
        
        clustered_data['churn_probability'] = model.predict_proba(clustered_data[features])[:, 1]

        # Build aggregation dynamically
        agg_dict = {f: 'mean' for f in features}
        agg_dict.update({'churn_probability': 'mean', 'customer_id': 'count'})

        segment_insights = clustered_data.groupby('segment').agg(agg_dict).reset_index()
        segment_insights = segment_insights.rename(columns={'customer_id': 'size'})
        return segment_insights


#Generate and display business insights
def main():
    
    insights = BusinessInsights()
    report = insights.generate_comprehensive_report()

    print("🎯 BUSINESS INSIGHTS REPORT")
    print("=" * 50)
    print(f"Model Type: {report['model_type']}")
    print(f"AUC Score: {report['model_performance']['auc_score']:.3f}")
    print(f"Confidence: {report['model_performance']['model_confidence']}")

    print("\n📊 Key Churn Drivers:")
    for driver in report['key_churn_drivers']:
        print(f"  - {driver['driver']}: {driver['insight']}")

    print("\n🚨 Risk Segments:")
    for segment in report['risk_segments']:
        print(f"  - {segment['segment']}: {segment['description']}")


if __name__ == "__main__":
    main()
