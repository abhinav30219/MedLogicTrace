"""
Mathematical reasoning dataset utilities for pretraining
"""

import json
import random
from typing import Dict, List, Tuple, Optional
from datasets import load_dataset
import re


class MathDatasetLoader:
    """Load and prepare mathematical reasoning datasets."""
    
    def __init__(self, dataset_name: str = "gsm8k", split: str = "train", max_samples: Optional[int] = None):
        self.dataset_name = dataset_name
        self.split = split
        self.dataset = None
        self.max_samples = max_samples
        
    def load_dataset(self) -> List[Dict]:
        """Load mathematical reasoning dataset."""
        if self.dataset_name == "gsm8k":
            # Load GSM8K dataset
            dataset = load_dataset("gsm8k", "main", split=self.split)
            
            processed_data = []
            for item in dataset:
                processed_data.append({
                    'question': item['question'],
                    'answer': item['answer'],
                    'reasoning_steps': self._extract_steps_from_gsm8k(item['answer']),
                    'final_answer': self._extract_final_answer(item['answer']),
                    'complexity': self._estimate_complexity(item['answer'])
                })
            
            return processed_data[:self.max_samples] if self.max_samples else processed_data
            
        elif self.dataset_name == "bespoke-stratos":
            # Load Bespoke-Stratos-17k dataset
            dataset = load_dataset("bespokelabs/Bespoke-Stratos-17k", split=self.split)
            
            processed_data = []
            # Randomly sample if max_samples is specified
            indices = list(range(len(dataset)))
            if self.max_samples and len(indices) > self.max_samples:
                indices = random.sample(indices, self.max_samples)
            
            for idx in indices:
                item = dataset[idx]
                # Extract problem and solution from the dataset format
                problem = item.get('problem', item.get('question', ''))
                solution = item.get('solution', item.get('answer', ''))
                
                processed_data.append({
                    'question': problem,
                    'answer': solution,
                    'reasoning_steps': self._extract_steps_from_solution(solution),
                    'final_answer': self._extract_final_answer_from_solution(solution),
                    'complexity': self._estimate_complexity(solution)
                })
            
            return processed_data
            
        elif self.dataset_name == "openr1-math":
            # Simulated OpenR1-Math dataset structure
            # In practice, you'd load from the actual dataset
            return self._generate_synthetic_math_data(self.max_samples or 1000)
        
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
    
    def _extract_steps_from_gsm8k(self, answer: str) -> List[str]:
        """Extract reasoning steps from GSM8K answer format."""
        # GSM8K answers contain step-by-step solutions
        lines = answer.strip().split('\n')
        steps = []
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith('####'):
                steps.append(line)
        
        return steps
    
    def _extract_final_answer(self, answer: str) -> str:
        """Extract final numerical answer."""
        # GSM8K format: #### [answer]
        match = re.search(r'####\s*(\d+)', answer)
        if match:
            return match.group(1)
        
        # Fallback: look for last number
        numbers = re.findall(r'\d+', answer)
        return numbers[-1] if numbers else ""
    
    def _extract_steps_from_solution(self, solution: str) -> List[str]:
        """Extract reasoning steps from a general solution format."""
        # Split by common step indicators
        lines = solution.strip().split('\n')
        steps = []
        
        for line in lines:
            line = line.strip()
            if line:
                # Clean up common prefixes
                line = re.sub(r'^(Step \d+:|Solution:|Therefore:|So,|Thus,)', '', line).strip()
                if line:
                    steps.append(line)
        
        return steps if steps else [solution.strip()]
    
    def _extract_final_answer_from_solution(self, solution: str) -> str:
        """Extract final answer from a general solution format."""
        # Look for common answer patterns
        patterns = [
            r'The answer is[:\s]+([^.]+)',
            r'Therefore[,:\s]+([^.]+)',
            r'Answer[:\s]+([^.]+)',
            r'= ([^.]+)$',
            r'####\s*(.+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, solution, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).strip()
        
        # Fallback: look for last number or expression
        lines = solution.strip().split('\n')
        if lines:
            last_line = lines[-1].strip()
            # Extract numbers or expressions from last line
            numbers = re.findall(r'[\d,]+\.?\d*', last_line)
            if numbers:
                return numbers[-1]
        
        return ""
    
    def _estimate_complexity(self, answer: str) -> int:
        """Estimate problem complexity based on solution length."""
        steps = self._extract_steps_from_gsm8k(answer)
        
        if len(steps) <= 2:
            return 1  # Simple
        elif len(steps) <= 4:
            return 2  # Medium
        else:
            return 3  # Complex
    
    def _generate_synthetic_math_data(self, num_samples: int) -> List[Dict]:
        """Generate synthetic mathematical reasoning data."""
        data = []
        
        templates = [
            # Simple arithmetic
            {
                'template': "What is {a} + {b}?",
                'solution': "To find {a} + {b}, I simply add the two numbers.\n{a} + {b} = {result}",
                'complexity': 1
            },
            # Multi-step arithmetic
            {
                'template': "If John has {a} apples and buys {b} more, then gives away {c}, how many does he have?",
                'solution': "Step 1: John starts with {a} apples.\nStep 2: He buys {b} more: {a} + {b} = {step1}\nStep 3: He gives away {c}: {step1} - {c} = {result}",
                'complexity': 2
            },
            # Word problem with multiplication
            {
                'template': "A store sells {a} boxes of cookies. Each box contains {b} cookies. If they sell all boxes, how many cookies did they sell?",
                'solution': "Step 1: Number of boxes = {a}\nStep 2: Cookies per box = {b}\nStep 3: Total cookies = {a} × {b} = {result}",
                'complexity': 2
            },
            # Complex multi-step
            {
                'template': "A factory produces {a} widgets per hour. They work {b} hours per day and {c} days per week. How many widgets do they produce in {d} weeks?",
                'solution': "Step 1: Widgets per hour = {a}\nStep 2: Hours per day = {b}, so widgets per day = {a} × {b} = {step1}\nStep 3: Days per week = {c}, so widgets per week = {step1} × {c} = {step2}\nStep 4: Number of weeks = {d}, so total widgets = {step2} × {d} = {result}",
                'complexity': 3
            }
        ]
        
        for _ in range(num_samples):
            template = random.choice(templates)
            
            # Generate random values
            a = random.randint(1, 100)
            b = random.randint(1, 100)
            c = random.randint(1, min(a + b, 50))
            d = random.randint(1, 10)
            
            # Calculate intermediate results
            if 'step1' in template['solution']:
                if '+' in template['solution']:
                    step1 = a + b
                elif '×' in template['solution']:
                    step1 = a * b
            else:
                step1 = 0
                
            if 'step2' in template['solution']:
                step2 = step1 * c
            else:
                step2 = 0
            
            # Calculate final result based on template
            if template['complexity'] == 1:
                result = a + b
            elif 'gives away' in template['template']:
                result = a + b - c
            elif 'boxes of cookies' in template['template']:
                result = a * b
            else:  # Factory problem
                result = a * b * c * d
            
            # Format question and solution
            question = template['template'].format(a=a, b=b, c=c, d=d)
            solution = template['solution'].format(
                a=a, b=b, c=c, d=d,
                step1=step1, step2=step2,
                result=result
            )
            
            data.append({
                'question': question,
                'answer': solution + f"\n#### {result}",
                'reasoning_steps': solution.split('\n'),
                'final_answer': str(result),
                'complexity': template['complexity']
            })
        
        return data
    
    def create_prompts(self, data: List[Dict], include_cot: bool = True) -> List[Dict]:
        """Create prompts for training."""
        prompts = []
        
        for item in data:
            if include_cot:
                # Chain-of-thought prompt
                prompt = f"Question: {item['question']}\n\nLet me solve this step by step.\n\n"
                reference = "\n".join(item['reasoning_steps']) + f"\n\nThe answer is {item['final_answer']}"
            else:
                # Direct answer prompt
                prompt = f"Question: {item['question']}\n\nAnswer: "
                reference = item['final_answer']
            
            prompts.append({
                'prompt': prompt,
                'reference': reference,
                'complexity': item['complexity'],
                'answer': item['final_answer']
            })
        
        return prompts


class MathToMedicalTransferDataset:
    """Dataset for progressive transfer from math to medical reasoning."""
    
    def __init__(self):
        self.stages = ['basic_medical', 'clinical_reasoning', 'complex_diagnosis']
        
    def create_transfer_curriculum(self) -> Dict[str, List[Dict]]:
        """Create progressive curriculum for transfer learning."""
        curriculum = {}
        
        # Stage 1: Basic medical calculations (similar to math)
        curriculum['basic_medical'] = self._create_basic_medical_problems()
        
        # Stage 2: Clinical reasoning with structure
        curriculum['clinical_reasoning'] = self._create_clinical_reasoning_problems()
        
        # Stage 3: Complex diagnostic reasoning
        curriculum['complex_diagnosis'] = self._create_diagnostic_problems()
        
        return curriculum
    
    def _create_basic_medical_problems(self) -> List[Dict]:
        """Create basic medical calculation problems."""
        problems = []
        
        templates = [
            {
                'question': "A patient needs {dose}mg of medication every {hours} hours. How much medication do they need in 24 hours?",
                'solution': "Step 1: Dose per administration = {dose}mg\nStep 2: Hours between doses = {hours}\nStep 3: Number of doses in 24 hours = 24 ÷ {hours} = {num_doses}\nStep 4: Total medication = {dose}mg × {num_doses} = {total}mg",
                'complexity': 2
            },
            {
                'question': "A patient's heart rate is {hr} beats per minute. How many times does their heart beat in {minutes} minutes?",
                'solution': "Step 1: Heart rate = {hr} beats/minute\nStep 2: Time period = {minutes} minutes\nStep 3: Total beats = {hr} × {minutes} = {total} beats",
                'complexity': 1
            },
            {
                'question': "A IV drip delivers {rate}mL per hour. The bag contains {volume}mL. How long will it last?",
                'solution': "Step 1: Drip rate = {rate}mL/hour\nStep 2: Total volume = {volume}mL\nStep 3: Duration = {volume}mL ÷ {rate}mL/hour = {duration} hours",
                'complexity': 2
            }
        ]
        
        for _ in range(100):
            template = random.choice(templates)
            
            # Generate values
            dose = random.choice([50, 100, 200, 250, 500])
            hours = random.choice([4, 6, 8, 12])
            hr = random.randint(60, 100)
            minutes = random.choice([5, 10, 15, 30])
            rate = random.choice([50, 100, 125, 150])
            volume = random.choice([500, 1000, 1500, 2000])
            
            # Calculate results
            if 'medication' in template['question']:
                num_doses = 24 // hours
                total = dose * num_doses
                solution = template['solution'].format(
                    dose=dose, hours=hours, num_doses=num_doses, total=total
                )
            elif 'heart rate' in template['question']:
                total = hr * minutes
                solution = template['solution'].format(
                    hr=hr, minutes=minutes, total=total
                )
            else:  # IV drip
                duration = volume / rate
                solution = template['solution'].format(
                    rate=rate, volume=volume, duration=duration
                )
            
            problems.append({
                'prompt': f"Question: {template['question'].format(dose=dose, hours=hours, hr=hr, minutes=minutes, rate=rate, volume=volume)}\n\nLet me solve this step by step.\n\n",
                'reference': solution,
                'complexity': template['complexity'],
                'stage': 'basic_medical'
            })
        
        return problems
    
    def _create_clinical_reasoning_problems(self) -> List[Dict]:
        """Create structured clinical reasoning problems."""
        problems = []
        
        cases = [
            {
                'presentation': "A 45-year-old patient presents with chest pain that worsens with deep breathing and improves when leaning forward.",
                'reasoning': "Step 1: Analyze symptoms - chest pain that is positional and pleuritic\nStep 2: Consider differential - pericarditis vs pleuritis vs musculoskeletal\nStep 3: Key finding - improvement with leaning forward is classic for pericarditis\nStep 4: Diagnosis - Acute pericarditis",
                'answer': "Acute pericarditis",
                'complexity': 2
            },
            {
                'presentation': "A patient has a blood pressure of 180/110 mmHg, severe headache, and blurred vision.",
                'reasoning': "Step 1: Identify vital signs - BP 180/110 (severely elevated)\nStep 2: Associated symptoms - headache and visual changes\nStep 3: This indicates end-organ damage\nStep 4: Diagnosis - Hypertensive emergency",
                'answer': "Hypertensive emergency",
                'complexity': 2
            }
        ]
        
        for case in cases:
            problems.append({
                'prompt': f"Clinical Case: {case['presentation']}\n\nAnalyze this case step by step.\n\n",
                'reference': case['reasoning'] + f"\n\nDiagnosis: {case['answer']}",
                'complexity': case['complexity'],
                'stage': 'clinical_reasoning'
            })
        
        return problems
    
    def _create_diagnostic_problems(self) -> List[Dict]:
        """Create complex diagnostic reasoning problems."""
        # Similar structure but more complex cases
        return []  # Implement as needed
