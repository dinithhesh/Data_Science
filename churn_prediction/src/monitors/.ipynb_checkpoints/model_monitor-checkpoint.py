"""
Model-Specific Monitoring and Performance Tracking
Monitors different metrics based on model type
"""

import json
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelMonitor:
    def __init__(self, config_path="configs/deployment_config.json"):
        self.config = self.load_config(config_path)
        self.model_type = None
        self.metrics_history = []
        
    def load_config(self, config_path):
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {"monitoring": {"interval_minutes": 60}}
    
    def load_model_info(self):
        """Load model type information"""
        try:
            with open("models/model_selection_results.json", "r") as f:
                results = json.load(f)
            self.model_type = results['best_model']
            logger.info(f"Monitoring setup for {self.model_type} model")
        except FileNotFoundError:
            logger.error("Model information not found")
            raise
    
    def get_model_specific_metrics(self):
        """Get model-specific metrics to monitor"""
        base_metrics = {
            'timestamp': datetime.now().isoformat(),
            'model_type': self.model_type,
            'throughput': 0,
            'latency_ms': 0,
            'error_rate': 0,
            'successful_requests': 0,
            'failed_requests': 0
        }
        
        if self.model_type == "RandomForest":
            # RF-specific metrics
            base_metrics.update({
                'avg_prediction_time': 0,
                'memory_usage_mb': 0,
                'feature_importance_stability': 1.0,
                'tree_depth_avg': 0,
                'oob_score': getattr(self, 'get_oob_score', lambda: None)()
            })
            
        elif self.model_type == "XGBoost":
            # XGBoost-specific metrics
            base_metrics.update({
                'avg_prediction_time': 0,
                'memory_usage_mb': 0,
                'feature_importance_stability': 1.0,
                'early_stopping_rounds': 0
            })
            
        elif self.model_type == "LogisticRegression":
            # LR-specific metrics
            base_metrics.update({
                'avg_prediction_time': 0,
                'memory_usage_mb': 0,
                'coefficient_stability': 1.0,
                'confidence_scores_avg': 0.5
            })
        
        return base_metrics
    
    def collect_metrics(self, service=None):
        """Collect current performance metrics"""
        self.load_model_info()
        metrics = self.get_model_specific_metrics()
        
        # Simulate metric collection (replace with actual monitoring)
        metrics.update({
            'throughput': np.random.randint(10, 100),
            'latency_ms': np.random.uniform(50, 500),
            'error_rate': np.random.uniform(0, 0.1),
            'successful_requests': np.random.randint(100, 1000),
            'failed_requests': np.random.randint(0, 10)
        })
        
        # Add model-specific metrics
        if self.model_type == "RandomForest":
            metrics.update({
                'avg_prediction_time': np.random.uniform(10, 100),
                'memory_usage_mb': np.random.randint(200, 500),
                'tree_depth_avg': np.random.randint(5, 20)
            })
        
        self.metrics_history.append(metrics)
        return metrics
    
    def detect_anomalies(self, metrics):
        """Detect anomalies based on model type"""
        thresholds = self.config.get('monitoring', {}).get('thresholds', {})
        
        anomalies = []
        
        # Common thresholds
        if metrics['error_rate'] > thresholds.get('max_error_rate', 0.05):
            anomalies.append(f"High error rate: {metrics['error_rate']:.3f}")
        
        if metrics['latency_ms'] > thresholds.get('max_latency_ms', 1000):
            anomalies.append(f"High latency: {metrics['latency_ms']:.1f}ms")
        
        # Model-specific anomaly detection
        if self.model_type == "RandomForest":
            if metrics.get('memory_usage_mb', 0) > thresholds.get('max_memory_mb', 1000):
                anomalies.append(f"High memory usage: {metrics['memory_usage_mb']}MB")
        
        elif self.model_type == "XGBoost":
            if metrics.get('avg_prediction_time', 0) > thresholds.get('max_prediction_time_ms', 200):
                anomalies.append(f"Slow predictions: {metrics['avg_prediction_time']:.1f}ms")
        
        return anomalies
    
    def generate_report(self, hours=24):
        """Generate monitoring report"""
        now = datetime.now()
        start_time = now - timedelta(hours=hours)
        
        recent_metrics = [
            m for m in self.metrics_history 
            if datetime.fromisoformat(m['timestamp']) >= start_time
        ]
        
        if not recent_metrics:
            return {"error": "No data available for the specified period"}
        
        df = pd.DataFrame(recent_metrics)
        
        report = {
            'report_period': f"Last {hours} hours",
            'model_type': self.model_type,
            'total_requests': df['successful_requests'].sum() + df['failed_requests'].sum(),
            'success_rate': df['successful_requests'].sum() / (df['successful_requests'].sum() + df['failed_requests'].sum()),
            'avg_latency_ms': df['latency_ms'].mean(),
            'avg_error_rate': df['error_rate'].mean(),
            'throughput_avg': df['throughput'].mean(),
            'anomalies_detected': len([a for m in recent_metrics for a in self.detect_anomalies(m)])
        }
        
        # Model-specific report sections
        if self.model_type == "RandomForest":
            report.update({
                'avg_memory_usage_mb': df['memory_usage_mb'].mean(),
                'avg_tree_depth': df['tree_depth_avg'].mean()
            })
        
        Path("monitoring").mkdir(exist_ok=True)
        report_path = f"monitoring/report_{now.strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📊 Monitoring report saved: {report_path}")
        return report
    
    def setup_alerts(self):
        """Setup model-specific alerting"""
        alerts = []
        
        if self.model_type == "RandomForest":
            alerts.extend([
                {"metric": "memory_usage_mb", "threshold": 800, "condition": "gt", "severity": "high"},
                {"metric": "error_rate", "threshold": 0.1, "condition": "gt", "severity": "critical"},
                {"metric": "latency_ms", "threshold": 2000, "condition": "gt", "severity": "high"}
            ])
        
        elif self.model_type == "XGBoost":
            alerts.extend([
                {"metric": "avg_prediction_time", "threshold": 500, "condition": "gt", "severity": "high"},
                {"metric": "error_rate", "threshold": 0.1, "condition": "gt", "severity": "critical"}
            ])
        
        elif self.model_type == "LogisticRegression":
            alerts.extend([
                {"metric": "error_rate", "threshold": 0.15, "condition": "gt", "severity": "critical"},
                {"metric": "confidence_scores_avg", "threshold": 0.3, "condition": "lt", "severity": "medium"}
            ])
        
        # Save alert configuration
        alert_config = {
            "model_type": self.model_type,
            "alerts": alerts,
            "setup_date": datetime.now().isoformat()
        }
        
        Path("monitoring").mkdir(exist_ok=True)
        with open("monitoring/alert_config.json", "w") as f:
            json.dump(alert_config, f, indent=2)
        
        logger.info(f"⚠️ Alert configuration saved for {self.model_type}")
        return alerts

def main():
    """Example monitoring setup"""
    monitor = ModelMonitor()
    monitor.load_model_info()
    
    # Collect sample metrics
    metrics = monitor.collect_metrics()
    print("Current metrics:", metrics)
    
    # Check for anomalies
    anomalies = monitor.detect_anomalies(metrics)
    if anomalies:
        print("Anomalies detected:", anomalies)
    
    # Setup alerts
    alerts = monitor.setup_alerts()
    print("Alerts configured:", alerts)
    
    # Generate report
    report = monitor.generate_report(hours=1)
    print("Report:", report)

if __name__ == "__main__":
    main()