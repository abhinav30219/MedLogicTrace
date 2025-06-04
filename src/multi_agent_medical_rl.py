"""
Multi-Agent RL System for Medical Reasoning
Implements specialized agents for diagnosis, treatment, verification, and efficiency
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataclasses import dataclass
import json


@dataclass
class AgentResponse:
    """Response from an individual agent."""
    agent_type: str
    response: str
    confidence: float
    reasoning_steps: List[str]
    tokens_used: int


class MedicalAgent:
    """Base class for specialized medical agents."""
    
    def __init__(self, model, tokenizer, agent_type: str):
        self.model = model
        self.tokenizer = tokenizer
        self.agent_type = agent_type
        self.temperature = 0.7
        self.max_length = 256
        
    def generate_response(self, prompt: str) -> AgentResponse:
        """Generate agent-specific response."""
        # Add agent-specific prompt engineering
        agent_prompt = self._format_prompt(prompt)
        
        # Generate
        inputs = self.tokenizer(agent_prompt, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.max_length,
                temperature=self.temperature,
                do_sample=True,
                top_p=0.95,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        response = self.tokenizer.decode(
            outputs.sequences[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )
        
        # Extract reasoning steps
        steps = self._extract_reasoning_steps(response)
        
        # Calculate confidence based on generation scores
        confidence = self._calculate_confidence(outputs.scores)
        
        # Count tokens
        tokens_used = len(outputs.sequences[0]) - inputs.input_ids.shape[1]
        
        return AgentResponse(
            agent_type=self.agent_type,
            response=response,
            confidence=confidence,
            reasoning_steps=steps,
            tokens_used=tokens_used
        )
    
    def _format_prompt(self, prompt: str) -> str:
        """Format prompt for specific agent type."""
        raise NotImplementedError
    
    def _extract_reasoning_steps(self, response: str) -> List[str]:
        """Extract reasoning steps from response."""
        import re
        steps = []
        
        # Look for numbered steps
        step_patterns = [
            r'(\d+[).])\s*([^\n]+)',
            r'Step\s*\d+:\s*([^\n]+)',
            r'(?:First|Second|Third|Next|Finally),\s*([^\n]+)'
        ]
        
        for pattern in step_patterns:
            matches = re.findall(pattern, response, re.MULTILINE)
            if matches:
                if isinstance(matches[0], tuple):
                    steps.extend([m[-1].strip() for m in matches])
                else:
                    steps.extend([m.strip() for m in matches])
                break
        
        # If no patterns found, split by sentences
        if not steps:
            sentences = re.split(r'[.!?]+', response)
            steps = [s.strip() for s in sentences if s.strip()]
        
        return steps
    
    def _calculate_confidence(self, scores) -> float:
        """Calculate confidence from generation scores."""
        if not scores:
            return 0.5
        
        # Average top-1 probability across tokens
        confidences = []
        for score in scores:
            probs = torch.softmax(score[0], dim=-1)
            top_prob = probs.max().item()
            confidences.append(top_prob)
        
        return np.mean(confidences)


class DiagnosticAgent(MedicalAgent):
    """Agent specialized in medical diagnosis."""
    
    def __init__(self, model, tokenizer):
        super().__init__(model, tokenizer, "diagnostic")
        self.temperature = 0.6  # Lower temperature for more focused diagnosis
        
    def _format_prompt(self, prompt: str) -> str:
        return f"""As a diagnostic specialist, analyze the following medical case:

{prompt}

Provide a systematic diagnostic approach:
1. Key symptoms and findings
2. Differential diagnosis
3. Most likely diagnosis
4. Reasoning for diagnosis

Focus on identifying the condition based on the presented information.
"""


class TreatmentAgent(MedicalAgent):
    """Agent specialized in treatment recommendations."""
    
    def __init__(self, model, tokenizer):
        super().__init__(model, tokenizer, "treatment")
        
    def _format_prompt(self, prompt: str) -> str:
        return f"""As a treatment specialist, consider the following case:

{prompt}

Provide evidence-based treatment recommendations:
1. Primary treatment approach
2. Medication if applicable
3. Non-pharmacological interventions
4. Follow-up care

Focus on practical, evidence-based interventions.
"""


class VerificationAgent(MedicalAgent):
    """Agent specialized in verifying medical reasoning."""
    
    def __init__(self, model, tokenizer):
        super().__init__(model, tokenizer, "verification")
        self.temperature = 0.5  # Lower temperature for critical analysis
        
    def _format_prompt(self, prompt: str) -> str:
        return f"""As a medical verification specialist, critically review:

{prompt}

Verify the medical reasoning by:
1. Checking logical consistency
2. Identifying potential errors or omissions
3. Confirming evidence-based practices
4. Flagging any concerns

Be thorough and critical in your analysis.
"""


class EfficiencyAgent(MedicalAgent):
    """Agent specialized in optimizing reasoning efficiency."""
    
    def __init__(self, model, tokenizer):
        super().__init__(model, tokenizer, "efficiency")
        self.max_length = 128  # Shorter responses for efficiency
        
    def _format_prompt(self, prompt: str) -> str:
        return f"""As an efficiency optimizer, provide the most concise medical analysis:

{prompt}

Provide only essential information:
1. Core finding
2. Key reasoning
3. Answer

Be extremely concise while maintaining accuracy.
"""


class MultiAgentMedicalSystem:
    """Coordinated multi-agent system for medical reasoning."""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-0.5B-Instruct",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        voting_threshold: float = 0.6,
        use_efficiency_agent: bool = True
    ):
        """
        Initialize multi-agent system.
        
        Args:
            model_name: Base model for all agents
            device: Device to run on
            voting_threshold: Consensus threshold
            use_efficiency_agent: Whether to include efficiency optimization
        """
        self.device = device
        self.voting_threshold = voting_threshold
        self.use_efficiency_agent = use_efficiency_agent
        
        # Load base model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize agents with shared model (in practice, could be different models)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        if device == "cuda":
            model = model.cuda()
        
        self.agents = {
            'diagnostic': DiagnosticAgent(model, self.tokenizer),
            'treatment': TreatmentAgent(model, self.tokenizer),
            'verification': VerificationAgent(model, self.tokenizer)
        }
        
        if use_efficiency_agent:
            self.agents['efficiency'] = EfficiencyAgent(model, self.tokenizer)
    
    def process_medical_case(
        self,
        case_description: str,
        return_all_responses: bool = False
    ) -> Dict:
        """
        Process a medical case through all agents.
        
        Args:
            case_description: Medical case to analyze
            return_all_responses: Whether to return all agent responses
            
        Returns:
            Dictionary with consensus results and metrics
        """
        # Phase 1: Diagnostic agent
        diag_response = self.agents['diagnostic'].generate_response(case_description)
        
        # Phase 2: Treatment agent (informed by diagnosis)
        treatment_prompt = f"{case_description}\n\nDiagnosis: {diag_response.response}"
        treatment_response = self.agents['treatment'].generate_response(treatment_prompt)
        
        # Phase 3: Verification agent reviews both
        verification_prompt = f"""{case_description}

Diagnostic Analysis:
{diag_response.response}

Treatment Recommendations:
{treatment_response.response}
"""
        verification_response = self.agents['verification'].generate_response(verification_prompt)
        
        # Phase 4: Efficiency optimization (optional)
        if self.use_efficiency_agent:
            efficiency_prompt = f"{case_description}\n\nProvide the most concise analysis based on the findings."
            efficiency_response = self.agents['efficiency'].generate_response(efficiency_prompt)
        else:
            efficiency_response = None
        
        # Build consensus
        consensus = self._build_consensus([
            diag_response,
            treatment_response,
            verification_response,
            efficiency_response
        ])
        
        # Calculate metrics
        total_tokens = sum(r.tokens_used for r in [diag_response, treatment_response, verification_response] if r)
        if efficiency_response:
            total_tokens += efficiency_response.tokens_used
        
        avg_confidence = np.mean([r.confidence for r in [diag_response, treatment_response, verification_response] if r])
        
        result = {
            'consensus_diagnosis': consensus['diagnosis'],
            'consensus_treatment': consensus['treatment'],
            'confidence_score': avg_confidence,
            'total_tokens_used': total_tokens,
            'verification_concerns': self._extract_concerns(verification_response.response),
            'efficient_summary': efficiency_response.response if efficiency_response else None
        }
        
        if return_all_responses:
            result['agent_responses'] = {
                'diagnostic': diag_response,
                'treatment': treatment_response,
                'verification': verification_response,
                'efficiency': efficiency_response
            }
        
        return result
    
    def _build_consensus(self, responses: List[AgentResponse]) -> Dict:
        """Build consensus from agent responses."""
        # Extract key information from responses
        diagnoses = []
        treatments = []
        
        for response in responses:
            if response and response.agent_type == 'diagnostic':
                # Extract diagnosis
                import re
                diag_match = re.search(r'diagnosis[:\s]+([^\n.]+)', response.response, re.IGNORECASE)
                if diag_match:
                    diagnoses.append(diag_match.group(1).strip())
            
            elif response and response.agent_type == 'treatment':
                # Extract treatment
                treat_match = re.search(r'treatment[:\s]+([^\n.]+)', response.response, re.IGNORECASE)
                if treat_match:
                    treatments.append(treat_match.group(1).strip())
        
        # Simple consensus: most mentioned
        consensus_diagnosis = diagnoses[0] if diagnoses else "Requires further evaluation"
        consensus_treatment = treatments[0] if treatments else "Supportive care pending diagnosis"
        
        return {
            'diagnosis': consensus_diagnosis,
            'treatment': consensus_treatment
        }
    
    def _extract_concerns(self, verification_response: str) -> List[str]:
        """Extract concerns from verification agent."""
        concerns = []
        
        # Look for concern indicators
        import re
        concern_patterns = [
            r'concern[:\s]+([^\n]+)',
            r'error[:\s]+([^\n]+)',
            r'missing[:\s]+([^\n]+)',
            r'should\s+(?:also\s+)?consider[:\s]+([^\n]+)'
        ]
        
        for pattern in concern_patterns:
            matches = re.findall(pattern, verification_response, re.IGNORECASE)
            concerns.extend([m.strip() for m in matches])
        
        return concerns
    
    def evaluate_consensus_quality(
        self,
        case_description: str,
        ground_truth: Optional[Dict] = None
    ) -> Dict:
        """Evaluate the quality of multi-agent consensus."""
        result = self.process_medical_case(case_description, return_all_responses=True)
        
        quality_metrics = {
            'agent_agreement': self._calculate_agreement(result['agent_responses']),
            'confidence_variance': self._calculate_confidence_variance(result['agent_responses']),
            'token_efficiency': self._calculate_token_efficiency(result),
            'reasoning_completeness': self._assess_reasoning_completeness(result['agent_responses'])
        }
        
        if ground_truth:
            quality_metrics['accuracy'] = self._compare_with_ground_truth(result, ground_truth)
        
        return quality_metrics
    
    def _calculate_agreement(self, agent_responses: Dict) -> float:
        """Calculate agreement score between agents."""
        # Simplified: check if key terms appear across responses
        responses = [r.response.lower() for r in agent_responses.values() if r]
        
        # Extract key medical terms
        all_terms = []
        for response in responses:
            import re
            # Extract medical-looking terms (simplified)
            terms = re.findall(r'\b[a-z]{4,}\b', response)
            all_terms.append(set(terms))
        
        # Calculate Jaccard similarity
        if len(all_terms) < 2:
            return 1.0
        
        intersection = all_terms[0]
        union = all_terms[0]
        for terms in all_terms[1:]:
            intersection = intersection.intersection(terms)
            union = union.union(terms)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _calculate_confidence_variance(self, agent_responses: Dict) -> float:
        """Calculate variance in agent confidence scores."""
        confidences = [r.confidence for r in agent_responses.values() if r]
        return np.var(confidences) if confidences else 0.0
    
    def _calculate_token_efficiency(self, result: Dict) -> float:
        """Calculate token efficiency metric."""
        # Compare full response tokens to efficient summary
        if result.get('efficient_summary') and result.get('agent_responses'):
            full_tokens = sum(r.tokens_used for r in result['agent_responses'].values() if r and r.agent_type != 'efficiency')
            efficient_tokens = result['agent_responses']['efficiency'].tokens_used if 'efficiency' in result['agent_responses'] else full_tokens
            return efficient_tokens / full_tokens if full_tokens > 0 else 1.0
        return 1.0
    
    def _assess_reasoning_completeness(self, agent_responses: Dict) -> float:
        """Assess completeness of reasoning across agents."""
        total_steps = 0
        for response in agent_responses.values():
            if response:
                total_steps += len(response.reasoning_steps)
        
        # Heuristic: good medical reasoning should have 10-20 total steps across agents
        completeness = min(total_steps / 15.0, 1.0)
        return completeness
    
    def _compare_with_ground_truth(self, result: Dict, ground_truth: Dict) -> float:
        """Compare consensus with ground truth."""
        score = 0.0
        
        if 'diagnosis' in ground_truth:
            if ground_truth['diagnosis'].lower() in result['consensus_diagnosis'].lower():
                score += 0.5
        
        if 'treatment' in ground_truth:
            if any(t in result['consensus_treatment'].lower() for t in ground_truth['treatment'].lower().split()):
                score += 0.5
        
        return score
