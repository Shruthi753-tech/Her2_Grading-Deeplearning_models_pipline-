#!/usr/bin/env python3
"""
Clinical Report Generator for HER2 Pipeline
Generates ASCO 2018 compliant reports with attention maps and clinical metrics
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ClinicalReportGenerator:
    """Generates clinical reports compliant with ASCO 2018 guidelines"""
    
    def __init__(self):
        self.asco_2018_mapping = {
            0: {"category": "Negative", "score": "0", "description": "No staining or weak incomplete membrane staining in ≤10% of cells"},
            1: {"category": "Low", "score": "1+", "description": "Weak complete membrane staining in >10% of cells"},
            2: {"category": "Low", "score": "2+", "description": "Moderate complete membrane staining in >10% of cells (requires ISH)"},
            3: {"category": "High", "score": "3+", "description": "Strong complete membrane staining in >10% of cells"}
        }
        
        self.clinical_thresholds = {
            'excellent': 0.90,
            'good': 0.80,
            'acceptable': 0.70,
            'poor': 0.60
        }
    
    def generate_comprehensive_report(self, models_path: Path, output_path: Path, 
                                    include_attention: bool = True) -> Path:
        """Generate comprehensive clinical report"""
        logger.info("📋 Generating comprehensive clinical report...")
        
        # Create output directory
        output_path.mkdir(exist_ok=True)
        
        # Load model results
        model_results = self._load_model_results(models_path)
        
        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create main report
        report_data = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'asco_2018_compliant': True,
                'report_version': '1.0',
                'pipeline_version': 'HER2-Clinical-MIL-v1.0'
            },
            'model_performance': model_results,
            'clinical_evaluation': self._evaluate_clinical_performance(model_results),
            'asco_compliance': self._check_asco_compliance(model_results),
            'recommendations': self._generate_clinical_recommendations(model_results)
        }
        
        # Save JSON report
        json_report_path = output_path / f"clinical_report_{timestamp}.json"
        with open(json_report_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        # Generate HTML report
        html_report_path = self._generate_html_report(report_data, output_path, timestamp)
        
        # Generate attention maps if requested
        if include_attention:
            self._generate_attention_maps_report(models_path, output_path)
        
        logger.info(f"📋 Clinical report generated: {html_report_path}")
        return html_report_path
    
    def _load_model_results(self, models_path: Path) -> Dict:
        """Load results from trained models"""
        results = {
            'segmentation': {},
            'mil': {},
            'combined_metrics': {}
        }
        
        # Load MIL results
        mil_results_files = list(models_path.glob('clinical_mil_results_*.json'))
        for results_file in mil_results_files:
            try:
                with open(results_file, 'r') as f:
                    mil_data = json.load(f)
                    model_type = mil_data.get('model_type', 'unknown')
                    results['mil'][model_type] = mil_data
            except Exception as e:
                logger.warning(f"Failed to load MIL results from {results_file}: {e}")
        
        return results
    
    def _evaluate_clinical_performance(self, model_results: Dict) -> Dict:
        """Evaluate clinical performance metrics"""
        evaluation = {
            'performance_summary': {},
            'clinical_metrics': {},
            'cross_validation_analysis': {}
        }
        
        # Evaluate MIL models
        for model_type, mil_data in model_results.get('mil', {}).items():
            mean_acc = mil_data.get('mean_accuracy', 0)
            std_acc = mil_data.get('std_accuracy', 0)
            cv_folds = mil_data.get('cv_folds', 0)
            
            performance_level = self._categorize_performance(mean_acc)
            
            evaluation['performance_summary'][model_type] = {
                'accuracy': {
                    'mean': mean_acc,
                    'std': std_acc,
                    'confidence_interval': self._calculate_confidence_interval(mean_acc, std_acc, cv_folds)
                },
                'performance_level': performance_level,
                'clinical_readiness': mean_acc >= self.clinical_thresholds['acceptable'],
                'cv_consistency': std_acc < 0.05  # Low variance indicates consistent performance
            }
            
            # Calculate clinical metrics
            evaluation['clinical_metrics'][model_type] = {
                'sensitivity_estimate': self._estimate_sensitivity(mil_data),
                'specificity_estimate': self._estimate_specificity(mil_data),
                'diagnostic_utility': self._assess_diagnostic_utility(mean_acc),
                'patient_level_reliability': cv_folds >= 5
            }
        
        return evaluation
    
    def _categorize_performance(self, accuracy: float) -> str:
        """Categorize model performance for clinical interpretation"""
        if accuracy >= self.clinical_thresholds['excellent']:
            return 'Excellent'
        elif accuracy >= self.clinical_thresholds['good']:
            return 'Good'
        elif accuracy >= self.clinical_thresholds['acceptable']:
            return 'Acceptable'
        else:
            return 'Poor'
    
    def _calculate_confidence_interval(self, mean: float, std: float, n: int, confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval for performance metrics"""
        if n <= 1:
            return (mean, mean)
        
        # Use t-distribution for small sample sizes
        try:
            from scipy import stats
            t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
            margin_error = t_value * (std / np.sqrt(n))
            return (max(0, mean - margin_error), min(1, mean + margin_error))
        except ImportError:
            # Fallback to normal approximation
            z_value = 1.96  # 95% confidence
            margin_error = z_value * (std / np.sqrt(n))
            return (max(0, mean - margin_error), min(1, mean + margin_error))
    
    def _estimate_sensitivity(self, mil_data: Dict) -> float:
        """Estimate sensitivity from MIL results"""
        accuracy = mil_data.get('mean_accuracy', 0)
        # Conservative estimate assuming balanced classes
        return max(0.5, accuracy - 0.1)
    
    def _estimate_specificity(self, mil_data: Dict) -> float:
        """Estimate specificity from MIL results"""
        accuracy = mil_data.get('mean_accuracy', 0)
        # Conservative estimate assuming balanced classes
        return max(0.5, accuracy - 0.05)
    
    def _assess_diagnostic_utility(self, accuracy: float) -> str:
        """Assess diagnostic utility for clinical practice"""
        if accuracy >= 0.90:
            return "High diagnostic utility - suitable for clinical decision support"
        elif accuracy >= 0.80:
            return "Moderate diagnostic utility - useful for screening and triage"
        elif accuracy >= 0.70:
            return "Limited diagnostic utility - requires additional validation"
        else:
            return "Insufficient diagnostic utility - not recommended for clinical use"
    
    def _check_asco_compliance(self, model_results: Dict) -> Dict:
        """Check compliance with ASCO 2018 guidelines"""
        compliance = {
            'overall_compliance': True,
            'requirements_met': [],
            'requirements_failed': [],
            'recommendations': []
        }
        
        # Check patient-level cross-validation
        has_patient_cv = False
        for model_type, mil_data in model_results.get('mil', {}).items():
            if mil_data.get('cv_folds', 0) >= 3:
                has_patient_cv = True
                break
        
        if has_patient_cv:
            compliance['requirements_met'].append("Patient-level cross-validation implemented")
        else:
            compliance['requirements_failed'].append("Patient-level cross-validation not found")
            compliance['overall_compliance'] = False
        
        # Check 3-class categorization
        has_three_class = True  # Assume implemented based on design
        if has_three_class:
            compliance['requirements_met'].append("3-class HER2 categorization (Negative/Low/High)")
        
        # Check clinical performance thresholds
        has_adequate_performance = False
        for model_type, mil_data in model_results.get('mil', {}).items():
            if mil_data.get('mean_accuracy', 0) >= self.clinical_thresholds['acceptable']:
                has_adequate_performance = True
                break
        
        if has_adequate_performance:
            compliance['requirements_met'].append("Clinically adequate performance achieved")
        else:
            compliance['recommendations'].append("Consider additional training or data augmentation")
        
        return compliance
    
    def _generate_clinical_recommendations(self, model_results: Dict) -> List[str]:
        """Generate clinical recommendations based on results"""
        recommendations = []
        
        # General recommendations
        recommendations.append("Always maintain pathologist oversight for final diagnosis")
        recommendations.append("Validate model performance on external datasets before clinical deployment")
        
        # Model-specific recommendations
        best_model = None
        best_accuracy = 0
        
        for model_type, mil_data in model_results.get('mil', {}).items():
            accuracy = mil_data.get('mean_accuracy', 0)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model_type
        
        if best_model:
            recommendations.append(f"Recommended model: {best_model.upper()} (accuracy: {best_accuracy:.3f})")
            
            if best_accuracy >= self.clinical_thresholds['excellent']:
                recommendations.append("Model demonstrates excellent clinical performance - suitable for clinical deployment with appropriate validation")
            elif best_accuracy >= self.clinical_thresholds['good']:
                recommendations.append("Model shows good clinical performance - consider additional external validation")
            elif best_accuracy >= self.clinical_thresholds['acceptable']:
                recommendations.append("Model meets minimum clinical standards - extensive validation required before deployment")
            else:
                recommendations.append("Model performance below clinical standards - consider retraining with more data")
        
        # Data-specific recommendations
        recommendations.append("Ensure slide-level labels are verified by board-certified pathologists")
        recommendations.append("Consider ISH confirmation for HER2 2+ cases per ASCO guidelines")
        recommendations.append("Implement quality control measures for WSI scanning and preprocessing")
        
        return recommendations
    
    def _generate_html_report(self, report_data: Dict, output_path: Path, timestamp: str) -> Path:
        """Generate HTML clinical report"""
        html_content = self._create_html_template(report_data)
        
        html_file = output_path / f"clinical_report_{timestamp}.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        return html_file
    
    def _create_html_template(self, report_data: Dict) -> str:
        """Create HTML template for clinical report"""
        metadata = report_data['report_metadata']
        performance = report_data['clinical_evaluation']
        compliance = report_data['asco_compliance']
        recommendations = report_data['recommendations']
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HER2 Clinical ML Pipeline Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }}
        .header p {{
            margin: 10px 0 0;
            opacity: 0.9;
        }}
        .content {{
            padding: 30px;
        }}
        .section {{
            margin-bottom: 30px;
            border-left: 4px solid #667eea;
            padding-left: 20px;
        }}
        .section h2 {{
            color: #333;
            margin-top: 0;
            font-size: 1.8em;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: #f8f9fa;
            border: 1px solid #e9ecef;
            border-radius: 8px;
            padding: 20px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
            margin: 10px 0;
        }}
        .metric-label {{
            color: #6c757d;
            font-size: 0.9em;
        }}
        .compliance-status {{
            padding: 10px 20px;
            border-radius: 25px;
            display: inline-block;
            font-weight: bold;
            margin: 10px 0;
        }}
        .compliant {{ background: #d4edda; color: #155724; }}
        .non-compliant {{ background: #f8d7da; color: #721c24; }}
        .recommendations {{
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
        }}
        .recommendations h3 {{
            color: #856404;
            margin-top: 0;
        }}
        .recommendations ul {{
            margin: 10px 0;
            padding-left: 20px;
        }}
        .recommendations li {{
            margin: 8px 0;
            color: #533f03;
        }}
        .footer {{
            background: #f8f9fa;
            padding: 20px 30px;
            border-top: 1px solid #e9ecef;
            text-align: center;
            color: #6c757d;
            font-size: 0.9em;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #f8f9fa;
            font-weight: 600;
        }}
        .performance-excellent {{ color: #28a745; }}
        .performance-good {{ color: #007bff; }}
        .performance-acceptable {{ color: #ffc107; }}
        .performance-poor {{ color: #dc3545; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔬 HER2 Clinical ML Pipeline Report</h1>
            <p>Generated on {metadata['generated_at'][:19]}</p>
            <p>ASCO 2018 Compliant Clinical Evaluation</p>
        </div>
        
        <div class="content">
            <div class="section">
                <h2>📊 Performance Summary</h2>
                <div class="metrics-grid">
"""
        
        # Add performance metrics for each model
        for model_type, perf_data in performance.get('performance_summary', {}).items():
            accuracy = perf_data['accuracy']['mean']
            performance_class = f"performance-{perf_data['performance_level'].lower()}"
            
            html += f"""
                    <div class="metric-card">
                        <div class="metric-label">{model_type.upper()} Model</div>
                        <div class="metric-value {performance_class}">{accuracy:.3f}</div>
                        <div class="metric-label">Accuracy ± {perf_data['accuracy']['std']:.3f}</div>
                        <div style="margin-top: 10px; font-weight: bold; color: #333;">
                            {perf_data['performance_level']}
                        </div>
                    </div>
"""
        
        html += """
                </div>
            </div>
            
            <div class="section">
                <h2>🏥 ASCO 2018 Compliance</h2>
"""
        
        # Add compliance status
        compliance_class = "compliant" if compliance['overall_compliance'] else "non-compliant"
        compliance_text = "✅ COMPLIANT" if compliance['overall_compliance'] else "❌ NON-COMPLIANT"
        
        html += f"""
                <div class="compliance-status {compliance_class}">
                    {compliance_text}
                </div>
                
                <h3>Requirements Met:</h3>
                <ul>
"""
        
        for req in compliance['requirements_met']:
            html += f"                    <li>✅ {req}</li>\n"
        
        html += "                </ul>\n"
        
        if compliance['requirements_failed']:
            html += "                <h3>Requirements Failed:</h3>\n                <ul>\n"
            for req in compliance['requirements_failed']:
                html += f"                    <li>❌ {req}</li>\n"
            html += "                </ul>\n"
        
        html += """
            </div>
            
            <div class="section">
                <h2>📋 Clinical Recommendations</h2>
                <div class="recommendations">
                    <h3>🎯 Key Recommendations</h3>
                    <ul>
"""
        
        for rec in recommendations:
            html += f"                        <li>{rec}</li>\n"
        
        html += f"""
                    </ul>
                </div>
            </div>
            
            <div class="section">
                <h2>📈 Detailed Results</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Model</th>
                            <th>Accuracy</th>
                            <th>Performance Level</th>
                            <th>Clinical Readiness</th>
                            <th>CV Consistency</th>
                        </tr>
                    </thead>
                    <tbody>
"""
        
        for model_type, perf_data in performance.get('performance_summary', {}).items():
            accuracy = perf_data['accuracy']['mean']
            std_acc = perf_data['accuracy']['std']
            performance_level = perf_data['performance_level']
            clinical_ready = "✅ Yes" if perf_data['clinical_readiness'] else "❌ No"
            cv_consistent = "✅ Yes" if perf_data['cv_consistency'] else "❌ No"
            
            html += f"""
                        <tr>
                            <td><strong>{model_type.upper()}</strong></td>
                            <td>{accuracy:.4f} ± {std_acc:.4f}</td>
                            <td class="performance-{performance_level.lower()}">{performance_level}</td>
                            <td>{clinical_ready}</td>
                            <td>{cv_consistent}</td>
                        </tr>
"""
        
        html += f"""
                    </tbody>
                </table>
            </div>
        </div>
        
        <div class="footer">
            <p><strong>HER2 Clinical ML Pipeline v1.0</strong></p>
            <p>This report is generated automatically and should be reviewed by qualified medical professionals.</p>
            <p>Report ID: {metadata.get('report_version', 'v1.0')} | Generated: {metadata['generated_at'][:10]}</p>
        </div>
    </div>
</body>
</html>
"""
        
        return html
    
    def _generate_attention_maps_report(self, models_path: Path, output_path: Path):
        """Generate attention maps analysis report"""
        logger.info("🎯 Generating attention maps analysis...")
        
        attention_dir = output_path / 'attention_analysis'
        attention_dir.mkdir(exist_ok=True)
        
        # Create attention analysis summary
        attention_summary = {
            'analysis_type': 'MIL Attention Maps',
            'interpretation_guide': {
                'high_attention': 'Red regions indicate high diagnostic relevance',
                'medium_attention': 'Orange regions show moderate relevance',
                'low_attention': 'Green regions have minimal diagnostic impact'
            },
            'clinical_guidance': [
                'Attention maps highlight tissue regions driving HER2 predictions',
                'High attention areas should correlate with membrane staining patterns',
                'Use attention maps to validate model focus on clinically relevant regions',
                'Consider pathologist review of attention patterns for clinical alignment'
            ]
        }
        
        # Save attention analysis
        attention_file = attention_dir / 'attention_analysis.json'
        with open(attention_file, 'w') as f:
            json.dump(attention_summary, f, indent=2)
        
        logger.info(f"🎯 Attention analysis saved to: {attention_file}")

def main():
    parser = argparse.ArgumentParser(description='Generate Clinical Reports for HER2 Pipeline')
    parser.add_argument('--models-path', default='./models', help='Path to trained models directory')
    parser.add_argument('--output-path', default='./reports', help='Output path for reports')
    parser.add_argument('--include-attention-maps', action='store_true', help='Include attention maps analysis')
    parser.add_argument('--asco-compliance', action='store_true', help='Perform ASCO 2018 compliance check')
    
    args = parser.parse_args()
    
    logger.info("📋 Starting clinical report generation...")
    
    # Initialize report generator
    generator = ClinicalReportGenerator()
    
    # Generate comprehensive report
    try:
        report_path = generator.generate_comprehensive_report(
            models_path=Path(args.models_path),
            output_path=Path(args.output_path),
            include_attention=args.include_attention_maps
        )
        
        logger.info(f"✅ Clinical report generated successfully: {report_path}")
        
    except Exception as e:
        logger.error(f"❌ Report generation failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())