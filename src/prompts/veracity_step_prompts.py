import os
import sys
from datetime import datetime
from typing import Dict, List, Literal


# project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)



################### STANCE #####################


def construct_prompt_stance(claim, evidence, few_shot=False):
    prompt = (
        "Classify the stance of the evidence toward the claim.\n"
        "Respond only with: Positive, Negative, or Neutral.\n"
        "Format: Stance: <label>\n\n"
    )
    if few_shot:
        prompt += "Examples:\n"
        prompt += "Claim 1: The Earth is round.\nEvidence 1: NASA has taken photographs from space and The Earth is round.\nStance 1: Positive\n"
        prompt += "Claim 2: Vaccines cause autism.\nEvidence 2: Multiple studies show there is no link between vaccines and autism.\nStance 2: Negative\n"
        prompt += "Claim 3: Aliens visited Earth in ancient times.\nEvidence 3: Some ancient drawings resemble modern technology.\nStance 3: Neutral\n\n"
    prompt += f"Claim: {claim}\nEvidence: {evidence}\nStance:"
    return prompt

'''
def construct_prompt_stance_cot2(claim, evidence, few_shot=False):
    print("Using CoT...\n")
    prompt = (
        "Determine the stance of the evidence with respect to the claim.\n"
        "Follow these steps to reason through your answer:\n"
        "1. Identify the main claim being made.\n"
        "2. Analyze what the provided evidence exactly states.\n"
        "3. Determine if the evidence:\n"
        "   - Directly SUPPORTS the claim (Positive)\n"
        "   - CONTRADICTS or REFUTES the claim (Negative)\n"
        "   - Neither clearly confirms nor refutes the claim (Neutral)\n"
        "4. Briefly justify your reasoning.\n"
        "5. Provide your final stance classification.\n\n"
        "Response format:\n"
        "Stance: [Positive/Negative/Neutral]\n\n"
    )
    prompt += f"Claim: {claim}\nEvidence: {evidence}\n\n"
    prompt += "Stance:"
    
    return prompt
    '''



def construct_prompt_stance_cot(claim, evidence, few_shot=False):
    print("Using CoT...\n")
    prompt = (
        "Determine the stance of the evidence with respect to the claim.\n"
        "Follow these steps to reason through your answer:\n"
        "1. Identify the main claim being made.\n"
        "2. Analyze what the provided evidence exactly states.\n"
        "3. Determine if the evidence:\n"
        "   - Directly SUPPORTS the claim (Positive)\n"
        "   - CONTRADICTS or REFUTES the claim (Negative)\n"
        "   - Neither clearly confirms nor refutes the claim (Neutral)\n"
        "4. Briefly justify your reasoning.\n"
        "5. Provide your final stance classification.\n\n"
        "Response format:\n"
        "Stance: [Positive/Negative/Neutral]\n\n"
    )
    
    if few_shot:
        prompt += "Examples:\n\n"
        
        # Example 1: Positive stance
        prompt += "Claim 1: The Earth is round.\n"
        prompt += "Evidence 1: NASA has taken photographs from space and The Earth is round.\n"
        prompt += "Analysis 1:\n"
        prompt += "1. The main claim is that the Earth is round.\n"
        prompt += "2. The evidence states that NASA has photographs from space showing the Earth is round.\n"
        prompt += "3. The evidence directly supports the claim by providing visual proof.\n"
        prompt += "4. The NASA photographs serve as direct confirmation of the Earth's roundness.\n"
        prompt += "5. Final classification: Positive\n"
        prompt += "Stance 1: Positive\n\n"
        
        # Example 2: Negative stance
        prompt += "Claim 2: Vaccines cause autism.\n"
        prompt += "Evidence 2: Multiple studies show there is no link between vaccines and autism.\n"
        prompt += "Analysis 2:\n"
        prompt += "1. The main claim is that vaccines cause autism.\n"
        prompt += "2. The evidence states that multiple studies show no link between vaccines and autism.\n"
        prompt += "3. The evidence directly contradicts the claim by showing no causal relationship.\n"
        prompt += "4. Scientific studies explicitly refute the claimed connection.\n"
        prompt += "5. Final classification: Negative\n"
        prompt += "Stance 2: Negative\n\n"
        
        # Example 3: Neutral stance
        prompt += "Claim 3: Aliens visited Earth in ancient times.\n"
        prompt += "Evidence 3: Some ancient drawings resemble modern technology.\n"
        prompt += "Analysis 3:\n"
        prompt += "1. The main claim is that aliens visited Earth in ancient times.\n"
        prompt += "2. The evidence mentions that some ancient drawings resemble modern technology.\n"
        prompt += "3. The evidence is ambiguous - resemblance doesn't prove alien visitation.\n"
        prompt += "4. The drawings could have multiple explanations and don't definitively support or refute the claim.\n"
        prompt += "5. Final classification: Neutral\n"
        prompt += "Stance 3: Neutral\n\n"
    
    prompt += f"Claim: {claim}\nEvidence: {evidence}\n\n"
    prompt += "Stance:"
    
    return prompt




################### VERACITY #####################


def construct_prompt_veracity(claim, veracity_labels, few_shot=False):
    prompt = (
        "Based on the stances, questions and evidence, classify the claim.\n"
        "Respond only with: Supported, Refuted, Not Enough Evidence, or Conflicting/Cherrypicking.\n"
        "Format: Veracity Prediction: <label>\n\n"
    )
    if few_shot:
        prompt += "Example 1:\nClaim 1: Wearing face masks will stop the spread of covid 19.\n"
        prompt += "Question 1: Does a face mask prevent the spread of Covid 19?"
        prompt += "Evidence 1: Cloth face coverings, even homemade masks made of the correct material, are effective in reducing the spread of COVID-19 - for the wearer and those around them - according to a new study from Oxford\u2019s Leverhulme Centre for Demographic Science.  Stance: Positive\n"
        prompt += "Veracity Prediction 1: Supported\n"

        prompt += "Example 2:\nClaim 2: Trump Administration claimed songwriter Billie Eilish Is Destroying Our Country In Leaked Documents\n"
        prompt += "Question 2: Has the Trump administration voiced that Billie Eilish was destroying the country?"        
        prompt += "Evidence 2: A Washington Post story wrongly claimed the Trump administration accused Billie Eilish of \u201cdestroying our country.  Stance: Negative\n"
        prompt += "Veracity Prediction 2: Refuted\n"

        prompt += "Example 3:\nClaim 3: It makes no sense for oil to be cheaper in Nigeria than in Saudi Arabia.\n"
        prompt += "Question 3: Why does fuel prices differ across countries?"        
        prompt += "Evidence 3: Fuel prices can be broken down into these major components.\n- fuel oil cost\n- refining cost\n- distribution cost, wholesale and retail\n- taxes, excise and VAT/sales tax. Stance: Neutral\n"
        prompt += "Veracity Prediction 3: Not Enough Evidence\n"
        

    prompt += f"Claim: {claim}\n"
    for idx, (question, evidence, stance) in enumerate(veracity_labels):
        prompt += f"Question {idx+1}: {question}\n"
        prompt += f"Evidence {idx+1}: {evidence}\nStance: {stance}\n"
    prompt += "Veracity Prediction:"
    return prompt



'''
def construct_prompt_veracity_cot2(claim, veracity_labels, few_shot=False):
    print("Using CoT...\n")
    prompt = (
        "Determine the veracity of the claim based on the provided questions, evidence, and stances.\n\n"
        "Follow these steps to reason through your answer:\n"
        "1. Analyze the main claim being made.\n"
        "2. Review each piece of evidence and its stance toward the claim.\n"
        "3. Consider the reliability and relevance of each piece of evidence.\n"
        "4. Determine if the overall evidence:\n"
        "   - SUPPORTS the claim (multiple reliable pieces of supporting evidence)\n"
        "   - REFUTES the claim (reliable evidence contradicting the claim)\n"
        "   - Provides NOT ENOUGH EVIDENCE to make a determination\n"
        "   - Shows CONFLICTING/CHERRYPICKING (evidence presents incomplete or contradictory picture)\n"
        "5. Justify your final veracity prediction classification with reasoning.\n\n"
        "Response format:\n"
        "Veracity Prediction: [Supported/Refuted/Not Enough Evidence/Conflicting/Cherrypicking]\n\n"
    )

    prompt += f"Claim: {claim}\n"
    for idx, (question, evidence, stance) in enumerate(veracity_labels):
        prompt += f"Question {idx+1}: {question}\n"
        prompt += f"Evidence {idx+1}: {evidence}\nStance: {stance}\n"
    prompt += "\nVeracity Prediction:"
    
    return prompt
'''


def construct_prompt_veracity_cot(claim, veracity_labels, few_shot=False):
    print("Using CoT...\n")
    prompt = (
        "Determine the veracity of the claim based on the provided questions, evidence, and stances.\n\n"
        "Follow these steps to reason through your answer:\n"
        "1. Analyze the main claim being made.\n"
        "2. Review each piece of evidence and its stance toward the claim.\n"
        "3. Consider the reliability and relevance of each piece of evidence.\n"
        "4. Determine if the overall evidence:\n"
        "   - SUPPORTS the claim (multiple reliable pieces of supporting evidence)\n"
        "   - REFUTES the claim (reliable evidence contradicting the claim)\n"
        "   - Provides NOT ENOUGH EVIDENCE to make a determination\n"
        "   - Shows CONFLICTING/CHERRYPICKING (evidence presents incomplete or contradictory picture)\n"
        "5. Justify your final veracity prediction classification with reasoning.\n\n"
        "Response format:\n"
        "Veracity Prediction: [Supported/Refuted/Not Enough Evidence/Conflicting/Cherrypicking]\n\n"
    )
    
    if few_shot:
        print("+ Few-Shot...\n")
        prompt += "Examples:\n\n"
        
        # Example 1: Supported
        prompt += "Example 1:\n"
        prompt += "Claim 1: Wearing face masks will stop the spread of covid 19.\n"
        prompt += "Question 1: Does a face mask prevent the spread of Covid 19?\n"
        prompt += "Evidence 1: Cloth face coverings, even homemade masks made of the correct material, are effective in reducing the spread of COVID-19 - for the wearer and those around them - according to a new study from Oxford's Leverhulme Centre for Demographic Science. Stance: Positive\n"
        prompt += "Analysis 1:\n"
        prompt += "1. The main claim is that wearing face masks will stop the spread of COVID-19.\n"
        prompt += "2. The evidence shows a positive stance, indicating masks are effective in reducing spread.\n"
        prompt += "3. The evidence comes from a credible source (Oxford study) and is directly relevant.\n"
        prompt += "4. The evidence supports the claim with reliable research findings.\n"
        prompt += "5. The Oxford study provides strong scientific backing for the effectiveness of masks.\n"
        prompt += "Veracity Prediction 1: Supported\n\n"
        
        # Example 2: Refuted
        prompt += "Example 2:\n"
        prompt += "Claim 2: Trump Administration claimed songwriter Billie Eilish Is Destroying Our Country In Leaked Documents\n"
        prompt += "Question 2: Has the Trump administration voiced that Billie Eilish was destroying the country?\n"
        prompt += "Evidence 2: A Washington Post story wrongly claimed the Trump administration accused Billie Eilish of \"destroying our country\". Stance: Negative\n"
        prompt += "Analysis 2:\n"
        prompt += "1. The main claim is that the Trump administration made statements about Billie Eilish destroying the country.\n"
        prompt += "2. The evidence has a negative stance, indicating the claim is false.\n"
        prompt += "3. The evidence is reliable (Washington Post correction) and directly addresses the claim.\n"
        prompt += "4. The evidence refutes the claim by showing it was based on a false report.\n"
        prompt += "5. The correction from a credible news source demonstrates the claim was unfounded.\n"
        prompt += "Veracity Prediction 2: Refuted\n\n"
        
        # Example 3: Not Enough Evidence
        prompt += "Example 3:\n"
        prompt += "Claim 3: It makes no sense for oil to be cheaper in Nigeria than in Saudi Arabia.\n"
        prompt += "Question 3: Why does fuel prices differ across countries?\n"
        prompt += "Evidence 3: Fuel prices can be broken down into these major components.\n- fuel oil cost\n- refining cost\n- distribution cost, wholesale and retail\n- taxes, excise and VAT/sales tax. Stance: Neutral\n"
        prompt += "Analysis 3:\n"
        prompt += "1. The main claim is about the logic of oil pricing differences between Nigeria and Saudi Arabia.\n"
        prompt += "2. The evidence has a neutral stance, providing general information about fuel price components.\n"
        prompt += "3. The evidence is relevant but doesn't specifically address the Nigeria vs Saudi Arabia comparison.\n"
        prompt += "4. The evidence doesn't provide enough information to determine if the claim makes sense or not.\n"
        prompt += "5. General fuel pricing factors don't specifically validate or invalidate the comparison claim.\n"
        prompt += "Veracity Prediction 3: Not Enough Evidence\n\n"
    
    prompt += f"Claim: {claim}\n"
    for idx, (question, evidence, stance) in enumerate(veracity_labels):
        prompt += f"Question {idx+1}: {question}\n"
        prompt += f"Evidence {idx+1}: {evidence}\nStance: {stance}\n"
    prompt += "\nVeracity Prediction:"
    
    return prompt






####### PYDANTIC PROMPTS ########

### STANCE ###

## zero-shot
def construct_prompt_stance_pydantic(claim: str, evidence: str, lang: str = "es", location: str = None) -> str:
    return f"""
You are tasked with determining the position of the evidence with respect to the claim.
Determine wether 'stance' is 'Positive', 'Negative', 'Neutral'.\n"

Here is the claim:
{claim}

Here is the evidence:
{evidence}

The JSON should include two main keys: "stance" and "reasoning". 

Your answer must be in {'Spanish' if lang == 'es' else 'English'}.
""".strip()


## few-shot
def construct_prompt_stance_pydantic_fewshot(claim: str, evidence: str, lang: str = "es", location: str = None) -> str:
    return f"""
Given a claim, you are tasked with determining the position of the evidence with respect to the claim.
Determine whether 'stance' is 'Positive', 'Negative', or 'Neutral'.

Examples:

Claim 1: The Earth is round.
Evidence 1: NASA has taken photographs from space and The Earth is round.
Stance 1: Positive
Reasoning 1: The evidence clearly supports the claim with direct observational data.

Claim 2: Vaccines cause autism.
Evidence 2: Multiple studies show there is no link between vaccines and autism.
Stance 2: Negative
Reasoning 2: The evidence contradicts the claim by citing studies that disprove it.

Claim 3: Aliens visited Earth in ancient times.
Evidence 3: Some ancient drawings resemble modern technology.
Stance 3: Neutral
Reasoning 3: The evidence is speculative and not conclusive enough to support or refute the claim.

Now analyze the following:

Claim 4: {claim}
Evidence 4: {evidence}

The JSON should include two main keys: "stance" and "reasoning".

Your answer must be in {'Spanish' if lang == 'es' else 'English'}.
""".strip()



#cot
def construct_prompt_stance_pydantic_cot(claim: str, evidence: str, lang: str = "es", location: str = None) -> str:
    return f"""
You are tasked with determining the position of the evidence with respect to the claim.
Determine whether 'stance' is 'Positive', 'Negative', 'Neutral'.

Here is the claim:
{claim}

Here is the evidence:
{evidence}

Follow this chain of thought process:

1. **Understanding**: First, carefully read and understand both the claim and the evidence.

2. **Key Elements Analysis**: Identify the main points, arguments, or assertions in both the claim and evidence.

3. **Relationship Assessment**: Analyze how the evidence relates to the claim:
   - Does the evidence support or validate the claim? (Positive)
   - Does the evidence contradict or refute the claim? (Negative) 
   - Does the evidence neither support nor contradict the claim, or is it unrelated? (Neutral)

4. **Supporting Details**: Consider specific phrases, data points, or arguments that justify your stance determination.

5. **Final Determination**: Based on your analysis, determine the overall stance.

The JSON should include two main keys: "stance" and "reasoning".

The "reasoning" field should include your step-by-step thought process covering:
- What the claim is asserting
- What the evidence presents
- How they relate to each other
- Why you chose the specific stance

Your answer must be in {'Spanish' if lang == 'es' else 'English'}.
""".strip()



def construct_prompt_stance_pydantic_cot_fewshot(claim: str, evidence: str, lang: str = "es", location: str = None) -> str:
    return f"""
You are tasked with determining the position of the evidence with respect to the claim.
Determine whether 'stance' is 'Positive', 'Negative', or 'Neutral'.

Examples with reasoning process:

Claim 1: The Earth is round.
Evidence 1: NASA has taken photographs from space and The Earth is round.

Chain of Thought 1:
1. **Understanding**: The claim states that the Earth is round. The evidence mentions NASA space photographs showing the Earth is round.
2. **Key Elements Analysis**: The claim asserts Earth's spherical shape; the evidence provides direct observational proof from space.
3. **Relationship Assessment**: The evidence directly supports the claim with photographic evidence from a credible source.
4. **Supporting Details**: NASA's space photographs provide visual confirmation of the Earth's roundness.
5. **Final Determination**: The evidence clearly validates the claim.

Stance 1: Positive
Reasoning 1: The evidence clearly supports the claim with direct observational data from NASA space photographs.

Claim 2: Vaccines cause autism.
Evidence 2: Multiple studies show there is no link between vaccines and autism.

Chain of Thought 2:
1. **Understanding**: The claim asserts vaccines cause autism. The evidence cites multiple studies finding no link.
2. **Key Elements Analysis**: The claim proposes a causal relationship; the evidence presents scientific research contradicting this.
3. **Relationship Assessment**: The evidence directly contradicts the claim by citing studies that disprove the connection.
4. **Supporting Details**: Multiple studies provide scientific evidence against the claimed causation.
5. **Final Determination**: The evidence refutes the claim with scientific research.

Stance 2: Negative
Reasoning 2: The evidence contradicts the claim by citing multiple studies that disprove the alleged vaccine-autism link.

Claim 3: Aliens visited Earth in ancient times.
Evidence 3: Some ancient drawings resemble modern technology.

Chain of Thought 3:
1. **Understanding**: The claim states aliens visited Earth in ancient times. The evidence mentions ancient drawings resembling modern technology.
2. **Key Elements Analysis**: The claim asserts extraterrestrial visitation; the evidence provides circumstantial visual similarities.
3. **Relationship Assessment**: The evidence is speculative and could have multiple explanations beyond alien visitation.
4. **Supporting Details**: Resemblance in drawings is not conclusive proof and could be coincidental or have other explanations.
5. **Final Determination**: The evidence is insufficient to support or refute the claim definitively.

Stance 3: Neutral
Reasoning 3: The evidence is speculative and not conclusive enough to support or refute the claim about ancient alien visitation.

Now analyze the following:

Claim 4: {claim}
Evidence 4: {evidence}

Follow this chain of thought process:

1. **Understanding**: First, carefully read and understand both the claim and the evidence.

2. **Key Elements Analysis**: Identify the main points, arguments, or assertions in both the claim and evidence.

3. **Relationship Assessment**: Analyze how the evidence relates to the claim:
   - Does the evidence support or validate the claim? (Positive)
   - Does the evidence contradict or refute the claim? (Negative) 
   - Does the evidence neither support nor contradict the claim, or is it unrelated? (Neutral)

4. **Supporting Details**: Consider specific phrases, data points, or arguments that justify your stance determination.

5. **Final Determination**: Based on your analysis, determine the overall stance.

The JSON should include two main keys: "stance" and "reasoning".

The "reasoning" field should include your step-by-step thought process covering:
- What the claim is asserting
- What the evidence presents
- How they relate to each other
- Why you chose the specific stance

Your answer must be in {'Spanish' if lang == 'es' else 'English'}.
""".strip()




#### VERACITY ###

## zero-shot
### no stance used
def construct_prompt_veracity__noStance_pydantic(claim: str, questions: List[str], evidences: List[str], lang: str = "es", location: str = None) -> str:
    prompt = f"Given a claim, you are tasked with establishing whether the evidences given 'Support', 'Refute', 'Not Enough Evidence' or 'Conflicting/Cherrypicking'.\n\n"
    prompt += f"For every claim there is one 'evidence' attribute which includes three 'answer' elements. Write one overall prediction for each claim in the 'pred_label' attribute in the json.\n"
    
    prompt += f"Taking into account all the 'answer' elements in 'evidence', you must predict the veracity towards the claim. \n"
    prompt += f"The veracity prediction must be one overall prediction for the whole 'evidence' attribute towards the claim, so there must only be one prediction label and reasoning per claim. \n"
    prompt += f"Write one overall prediction for the claim in the 'pred_label' attribute in the json and provide the 'reasoning'.\n"
    
    prompt += f"Claim: {claim}\n\n"

    for idx, (question, evidence) in enumerate(zip(questions, evidences)):
        prompt += f"Question {idx+1}: {question}\n"
        prompt += f"Evidence {idx+1}: {evidence}\n\n"

    prompt += (
        "The JSON should include two main keys: \"pred_label\" and \"reasoning\".\n\n"
        f"Your answer must be in {'Spanish' if lang == 'es' else 'English'}."
    )

    return prompt.strip()


### stance used
def construct_prompt_veracity__withStance_pydantic(claim: str, questions: List[str], evidences: List[str], stances: List[str], lang: str = "es", location: str = None) -> str:
    prompt = f"Given a claim, you are tasked with establishing whether the evidences given 'Support', 'Refute', 'Not Enough Evidence' or 'Conflicting/Cherrypicking'.\n\n"
    prompt += f"For every claim there is one 'evidence' attribute which includes three 'answer' elements and their respective 'stance' elements. \n"
    
    prompt += f"Taking into account all the 'answer' elements in 'evidence', you must predict the veracity towards the claim. \n"
    prompt += f"The veracity prediction must be one overall prediction for the whole 'evidence' attribute towards the claim, so there must only be one prediction label and reasoning per claim. \n"
    prompt += f"Write one overall prediction for the claim in the 'pred_label' attribute in the json and provide the 'reasoning'.\n"
    
    
    prompt += f"Claim: {claim}\n\n"

    for idx, (question, evidence, stance) in enumerate(zip(questions, evidences, stances)):
        prompt += f"Question {idx+1}: {question}\n"
        prompt += f"Evidence {idx+1}: {evidence}\n\n"
        prompt += f"Stance {idx+1}: {stance}\n\n"

    prompt += (
        "The JSON should include two main keys: \"pred_label\" and \"reasoning\".\n\n"
        f"Your answer must be in {'Spanish' if lang == 'es' else 'English'}."
    )

    return prompt.strip()

## few-shot
### no stance used
def construct_prompt_veracity__noStance_pydantic_fewshot(claim: str, questions: List[str], evidences: List[str], lang: str = "es", location: str = None) -> str:
    prompt = f"""
Given a claim, you are tasked with establishing whether the evidences given 'Support', 'Refute', 'Not Enough Evidence' or 'Conflicting/Cherrypicking'.\n\n"
For every claim there is one 'evidence' attribute which includes three 'answer' elements. 
Taking into account all the 'answer' elements in 'evidence', you must predict the veracity towards the claim. 
The veracity prediction must be one overall prediction for the whole 'evidence' attribute towards the claim, so there must only be one prediction label and reasoning per claim. 
Write one overall prediction for the claim in the 'pred_label' attribute in the json and provide the 'reasoning'.\n"
    
You will be given some examples first. 

Examples:

Claim 1: Wearing face masks will stop the spread of COVID-19.
Question 1: Does a face mask prevent the spread of COVID-19?
Evidence 1: Cloth face coverings, even homemade ones, are effective in reducing the spread of COVID-19, according to Oxford's Leverhulme Centre.
Veracity Prediction 1: Supported
Reasoning 1: The evidence confirms the claim through a credible study.

Claim 2: Trump Administration claimed Billie Eilish is destroying the country.
Question 2: Has the Trump administration made that claim?
Evidence 2: A Washington Post article wrongly stated this; no official documents confirm the claim.
Veracity Prediction 2: Refuted
Reasoning 2: The claim is explicitly debunked by the evidence.

Claim 3: It makes no sense for oil to be cheaper in Nigeria than in Saudi Arabia.
Question 3: Why do fuel prices differ by country?
Evidence 3: Fuel prices vary due to taxes, refining costs, and other components.
Veracity Prediction 3: Not Enough Evidence
Reasoning 3: The evidence provides general background but does not evaluate the specific comparison in the claim.

Now analyze the following:

Claim 4: {claim}
"""
    for idx, (question, evidence) in enumerate(zip(questions, evidences)):
        prompt += f"\nQuestion 4.{idx+1}: {question}\nEvidence 4.{idx+1}: {evidence}"
    prompt += (
        "\n\nThe JSON should include two main keys: \"pred_label\" and \"reasoning\".\n"
        f"Your answer must be in {'Spanish' if lang == 'es' else 'English'}."
    )
    return prompt.strip()





### stance used
def construct_prompt_veracity__withStance_pydantic_fewshot(claim: str, questions: List[str], evidences: List[str], stances: List[str], lang: str = "es", location: str = None) -> str:
    prompt = f"""
Given a claim, you are tasked with establishing whether the evidences given 'Support', 'Refute', 'Not Enough Evidence' or 'Conflicting/Cherrypicking'.\n\n"
For every claim there is one 'evidence' attribute which includes three 'answer' and their respective 'stance' elements. 
Taking into account all the 'answer' elements in 'evidence', you must predict the veracity towards the claim. 
The veracity prediction must be one overall prediction for the whole 'evidence' attribute towards the claim, so there must only be one prediction label and reasoning per claim. 
Write one overall prediction for the claim in the 'pred_label' attribute in the json and provide the 'reasoning'.\n"
    
You will be given some examples first. 

Examples:

Claim 1: Wearing face masks will stop the spread of COVID-19.
Question 1: Does a face mask prevent the spread of COVID-19?
Evidence 1: Cloth face coverings, even homemade ones, are effective in reducing the spread of COVID-19, according to Oxford's Leverhulme Centre.
Stance 1: Positive
Veracity Prediction 1: Supported
Reasoning 1: The evidence confirms the claim through a credible study.

Claim 2: Trump Administration claimed Billie Eilish is destroying the country.
Question 2: Has the Trump administration made that claim?
Evidence 2: A Washington Post article wrongly stated this; no official documents confirm the claim.
Stance 2: Negative
Veracity Prediction 2: Refuted
Reasoning 2: The claim is explicitly debunked by the evidence.

Claim 3: It makes no sense for oil to be cheaper in Nigeria than in Saudi Arabia.
Question 3: Why do fuel prices differ by country?
Evidence 3: Fuel prices vary due to taxes, refining costs, and other components.
Stance 3: Neutral
Veracity Prediction 3: Not Enough Evidence
Reasoning 3: The evidence provides general background but does not evaluate the specific comparison in the claim.

Now analyze the following:

Claim 4: {claim}
"""
    for idx, (question, evidence, stance) in enumerate(zip(questions, evidences, stances)):
        prompt += f"\nQuestion 4.{idx+1}: {question}"
        prompt += f"\nEvidence 4.{idx+1}: {evidence}"
        prompt += f"\nStance 4.{idx+1}: {stance}"
    prompt += (
        "\n\nThe JSON should include two main keys: \"pred_label\" and \"reasoning\".\n"
        f"Your answer must be in {'Spanish' if lang == 'es' else 'English'}."
    )
    return prompt.strip()




# CoT:
# clauderekin 
# gpt4-k ere prompt generation

## adibidea:
# no stance
def construct_prompt_veracity__noStance_pydantic_cot(claim: str, questions: List[str], evidences: List[str], lang: str = "es", location: str = None) -> str:
    """
    Constructs a chain of thought prompt for veracity assessment that guides the model
    through systematic reasoning before making a final prediction.
    """
    language_instruction = "Spanish" if lang == "es" else "English"
    prompt = f"""Given a claim, you are tasked with establishing whether the evidences given 'Supported', 'Refuted', 'Not Enough Evidence' or 'Conflicting/Cherrypicking'.
For every claim there is one 'evidence' attribute which includes multiple 'answer' elements. You must provide one overall prediction for each claim in the 'pred_label' attribute in the JSON.
**REASONING PROCESS - Follow these steps systematically:**
1. **CLAIM ANALYSIS**: First, identify the key assertions and components of the claim that need to be verified.
2. **EVIDENCE EXAMINATION**: For each piece of evidence provided:
   - Summarize what the evidence states
   - Assess the quality and reliability of the evidence
   - Determine how directly it relates to the claim
3. **INDIVIDUAL EVIDENCE ASSESSMENT**: For each evidence piece, determine if it:
   - Supports the claim (and to what degree)
   - Refutes the claim (and to what degree)
   - Is neutral/irrelevant to the claim
   - Contains conflicting information
4. **PATTERN ANALYSIS**: Look across all evidence pieces for:
   - Consistent patterns of support or refutation
   - Contradictions between different evidence sources
   - Gaps in information needed to verify the claim
   - Signs of selective evidence presentation (cherry-picking)
5. **SYNTHESIS**: Combine your analysis of all evidence to determine the overall relationship between the evidence set and the claim.
6. **FINAL CLASSIFICATION**: Based on your analysis, classify as:
   - **Supported**: Evidence consistently and reliably supports the claim
   - **Refuted**: Evidence consistently and reliably contradicts the claim
   - **Not Enough Evidence**: Insufficient reliable evidence to make a determination
   - **Conflicting/Cherrypicking**: Evidence presents contradictory information or appears selectively chosen
**CLAIM TO EVALUATE:**
{claim}
**EVIDENCE TO ANALYZE:**
"""
    # Add questions and evidences
    for idx, (question, evidence) in enumerate(zip(questions, evidences)):
        prompt += f"""
Question {idx+1}: {question}
Evidence {idx+1}: {evidence}
"""
    prompt += f"""
**INSTRUCTIONS FOR YOUR RESPONSE:**
Please work through each step of the reasoning process outlined above. Show your thinking clearly for each step before providing your final answer.
Taking into account all the 'answer' elements in 'evidence', you must predict the veracity towards the claim. The veracity prediction must be one overall prediction for the whole 'evidence' attribute towards the claim, so there must only be one prediction label and reasoning per claim.
Provide your response as a JSON with the following structure:
{{
    "pred_label": " 'Supported' / 'Refuted' / 'Not Enough Evidence' / 'Conflicting/Cherrypicking' ",
    "reasoning": "Your comprehensive reasoning for the final classification"
}}
Your answer must be in {language_instruction}."""
    return prompt.strip()



def construct_prompt_veracity__withStance_pydantic_cot(claim: str, questions: List[str], evidences: List[str], stances: List[str], lang: str = "es", location: str = None) -> str:
    """
    Constructs a chain of thought prompt for veracity assessment that guides the model
    through systematic reasoning before making a final prediction.
    """
    language_instruction = "Spanish" if lang == "es" else "English"
    prompt = f"""Given a claim, you are tasked with establishing whether the evidences given 'Supported', 'Refuted', 'Not Enough Evidence' or 'Conflicting/Cherrypicking'.
For every claim there is one 'evidence' attribute which includes multiple 'answer' elements. You must provide one overall prediction for each claim in the 'pred_label' attribute in the JSON.
**REASONING PROCESS - Follow these steps systematically:**
1. **CLAIM ANALYSIS**: First, identify the key assertions and components of the claim that need to be verified.
2. **EVIDENCE EXAMINATION**: For each piece of evidence provided:
   - Summarize what the evidence states
   - Assess the quality and reliability of the evidence
   - Determine how directly it relates to the claim
3. **INDIVIDUAL EVIDENCE ASSESSMENT**: For each evidence piece, determine if it:
   - Supports the claim (and to what degree)
   - Refutes the claim (and to what degree)
   - Is neutral/irrelevant to the claim
   - Contains conflicting information
4. **PATTERN ANALYSIS**: Look across all evidence pieces for:
   - Consistent patterns of support or refutation
   - Contradictions between different evidence sources
   - Gaps in information needed to verify the claim
   - Signs of selective evidence presentation (cherry-picking)
5. **SYNTHESIS**: Combine your analysis of all evidence to determine the overall relationship between the evidence set and the claim.
6. **FINAL CLASSIFICATION**: Based on your analysis, classify as:
   - **Supported**: Evidence consistently and reliably supports the claim
   - **Refuted**: Evidence consistently and reliably contradicts the claim
   - **Not Enough Evidence**: Insufficient reliable evidence to make a determination
   - **Conflicting/Cherrypicking**: Evidence presents contradictory information or appears selectively chosen
**CLAIM TO EVALUATE:**
{claim}
**EVIDENCE TO ANALYZE:**
"""
    # Add questions and evidences
    for idx, (question, evidence, stance) in enumerate(zip(questions, evidences, stances)):
        prompt += f"""
Question {idx+1}: {question}
Evidence {idx+1}: {evidence}
Stance {idx+1}: {stance}
"""
    prompt += f"""
**INSTRUCTIONS FOR YOUR RESPONSE:**
Please work through each step of the reasoning process outlined above. Show your thinking clearly for each step before providing your final answer.
Taking into account all the 'answer' elements in 'evidence', you must predict the veracity towards the claim. The veracity prediction must be one overall prediction for the whole 'evidence' attribute towards the claim, so there must only be one prediction label and reasoning per claim.
Provide your response as a JSON with the following structure:
{{
    "pred_label": " 'Supported' / 'Refuted' / 'Not Enough Evidence' / 'Conflicting/Cherrypicking' ",
    "reasoning": "Your comprehensive reasoning for the final classification"
}}
Your answer must be in {language_instruction}."""
    return prompt.strip()




# cot + fewshot
# no stance
def construct_prompt_veracity__noStance_pydantic_cot_fewshot(claim: str, questions: List[str], evidences: List[str], lang: str = "es", location: str = None) -> str:
    """
    Constructs a combined chain of thought and few-shot prompt for veracity assessment
    that provides examples with detailed reasoning and guides systematic analysis.
    """
    language_instruction = "Spanish" if lang == "es" else "English"
    
    prompt = f"""Given a claim, you are tasked with establishing whether the evidences given 'Supported', 'Refuted', 'Not Enough Evidence' or 'Conflicting/Cherrypicking'.

For every claim there is one 'evidence' attribute which includes multiple 'answer' elements. You must provide one overall prediction for each claim in the 'pred_label' attribute in the JSON.

**EXAMPLES WITH DETAILED REASONING:**

**Example 1:**
Claim 1: Wearing face masks will stop the spread of COVID-19.
Question 1: Does a face mask prevent the spread of COVID-19?
Evidence 1: Cloth face coverings, even homemade ones, are effective in reducing the spread of COVID-19, according to Oxford's Leverhulme Centre.

**Chain of Thought Analysis 1:**
1. **CLAIM ANALYSIS**: The claim asserts that face masks will stop COVID-19 spread.
2. **EVIDENCE EXAMINATION**: The evidence cites a credible Oxford study showing masks are effective in reducing spread.
3. **INDIVIDUAL EVIDENCE ASSESSMENT**: The evidence strongly supports the claim with academic research.
4. **PATTERN ANALYSIS**: Single piece of evidence from reliable source consistently supports the claim.
5. **SYNTHESIS**: The Oxford study provides strong scientific backing for mask effectiveness.
6. **FINAL CLASSIFICATION**: The evidence reliably supports the claim.

Veracity Prediction 1: Support
Reasoning 1: The evidence from Oxford's Leverhulme Centre provides credible scientific support for the effectiveness of face masks in reducing COVID-19 spread.

**Example 2:**
Claim 2: Trump Administration claimed Billie Eilish is destroying the country.
Question 2: Has the Trump administration made that claim?
Evidence 2: A Washington Post article wrongly stated this; no official documents confirm the claim.

**Chain of Thought Analysis 2:**
1. **CLAIM ANALYSIS**: The claim asserts the Trump administration made specific statements about Billie Eilish.
2. **EVIDENCE EXAMINATION**: The evidence shows a Washington Post correction indicating the claim was false.
3. **INDIVIDUAL EVIDENCE ASSESSMENT**: The evidence directly refutes the claim by showing it was based on incorrect reporting.
4. **PATTERN ANALYSIS**: Single piece of evidence from reliable source consistently refutes the claim.
5. **SYNTHESIS**: The correction from a credible news source demonstrates the claim was unfounded.
6. **FINAL CLASSIFICATION**: The evidence reliably contradicts the claim.

Veracity Prediction 2: Refute
Reasoning 2: The evidence explicitly debunks the claim by showing it was based on incorrect reporting, with no official documents supporting the alleged statement.

**Example 3:**
Claim 3: It makes no sense for oil to be cheaper in Nigeria than in Saudi Arabia.
Question 3: Why do fuel prices differ by country?
Evidence 3: Fuel prices vary due to taxes, refining costs, and other components.

**Chain of Thought Analysis 3:**
1. **CLAIM ANALYSIS**: The claim makes a judgment about the logic of oil pricing differences between two specific countries.
2. **EVIDENCE EXAMINATION**: The evidence provides general information about fuel price components but doesn't address the specific comparison.
3. **INDIVIDUAL EVIDENCE ASSESSMENT**: The evidence is relevant but doesn't directly evaluate the Nigeria vs Saudi Arabia comparison.
4. **PATTERN ANALYSIS**: Single piece of evidence provides background information but lacks specific analysis needed.
5. **SYNTHESIS**: General pricing factors don't address the specific claim about these two countries.
6. **FINAL CLASSIFICATION**: Insufficient evidence to determine the validity of the specific comparison.

Veracity Prediction 3: Not Enough Evidence
Reasoning 3: The evidence provides general background on fuel pricing factors but does not specifically evaluate the comparison between Nigeria and Saudi Arabia mentioned in the claim.

**REASONING PROCESS - Follow these steps systematically:**

1. **CLAIM ANALYSIS**: First, identify the key assertions and components of the claim that need to be verified.

2. **EVIDENCE EXAMINATION**: For each piece of evidence provided:
   - Summarize what the evidence states
   - Assess the quality and reliability of the evidence
   - Determine how directly it relates to the claim

3. **INDIVIDUAL EVIDENCE ASSESSMENT**: For each evidence piece, determine if it:
   - Supports the claim (and to what degree)
   - Refutes the claim (and to what degree)
   - Is neutral/irrelevant to the claim
   - Contains conflicting information

4. **PATTERN ANALYSIS**: Look across all evidence pieces for:
   - Consistent patterns of support or refutation
   - Contradictions between different evidence sources
   - Gaps in information needed to verify the claim
   - Signs of selective evidence presentation (cherry-picking)

5. **SYNTHESIS**: Combine your analysis of all evidence to determine the overall relationship between the evidence set and the claim.

6. **FINAL CLASSIFICATION**: Based on your analysis, classify as:
   - **Supported**: Evidence consistently and reliably supports the claim
   - **Refuted**: Evidence consistently and reliably contradicts the claim
   - **Not Enough Evidence**: Insufficient reliable evidence to make a determination
   - **Conflicting/Cherrypicking**: Evidence presents contradictory information or appears selectively chosen

**CLAIM TO EVALUATE:**
{claim}

**EVIDENCE TO ANALYZE:**"""
    
    # Add questions and evidences
    for idx, (question, evidence) in enumerate(zip(questions, evidences)):
        prompt += f"""
Question {idx+1}: {question}
Evidence {idx+1}: {evidence}"""
    
    prompt += f"""

**INSTRUCTIONS FOR YOUR RESPONSE:**
Please work through each step of the reasoning process outlined above. Show your thinking clearly for each step before providing your final answer.

Taking into account all the 'answer' elements in 'evidence', you must predict the veracity towards the claim. The veracity prediction must be one overall prediction for the whole 'evidence' attribute towards the claim, so there must only be one prediction label and reasoning per claim.

Provide your response as a JSON with the following structure:
{{
    "pred_label": " 'Supported' / 'Refuted' / 'Not Enough Evidence' / 'Conflicting/Cherrypicking' ",
    "reasoning": "Your comprehensive reasoning for the final classification"
}}

Your answer must be in {language_instruction}."""
    
    return prompt.strip()



# with stance

def construct_prompt_veracity__withStance_pydantic_cot_fewshot2(claim: str, questions: List[str], evidences: List[str], stances: List[str], lang: str = "es", location: str = None) -> str:
    """
    Constructs a combined chain of thought and few-shot prompt for veracity assessment with stance information
    that provides examples with detailed reasoning and guides systematic analysis.
    """
    language_instruction = "Spanish" if lang == "es" else "English"
    
    prompt = f"""Given a claim, you are tasked with establishing whether the evidences given 'Supported', 'Refuted', 'Not Enough Evidence' or 'Conflicting/Cherrypicking'.

    """
    
    # Add questions, evidences, and stances
    for idx, (question, evidence, stance) in enumerate(zip(questions, evidences, stances)):
        prompt += f"""
    Question {idx+1}: {question}
    Evidence {idx+1}: {evidence}
    Stance {idx+1}: {stance}"""
        
        prompt += f"""

    **INSTRUCTIONS FOR YOUR RESPONSE:**
    Please work through each step of the reasoning process outlined above. Show your thinking clearly for each step before providing your final answer.

    Taking into account all the 'answer' elements in 'evidence' and their respective stances, you must predict the veracity towards the claim. The veracity prediction must be one overall prediction for the whole 'evidence' attribute towards the claim, so there must only be one prediction label and reasoning per claim.

    Provide your response as a JSON with the following structure:
    {{
        "pred_label": " 'Supported' / 'Refuted' / 'Not Enough Evidence' / 'Conflicting/Cherrypicking' ",
        "reasoning": "Your comprehensive reasoning for the final classification"
    }}
    Your answer must be in {language_instruction}."""

    return prompt.strip()
    




def construct_prompt_veracity__withStance_pydantic_cot_fewshot(claim: str, questions: List[str], evidences: List[str], stances: List[str], lang: str = "es", location: str = None) -> str:
    """
    Constructs a combined chain of thought and few-shot prompt for veracity assessment with stance information
    that provides examples with detailed reasoning and guides systematic analysis.
    """
    language_instruction = "Spanish" if lang == "es" else "English"
    
    prompt = f"""Given a claim, you are tasked with establishing whether the evidences given 'Supported', 'Refuted', 'Not Enough Evidence' or 'Conflicting/Cherrypicking'.

For every claim there is one 'evidence' attribute which includes multiple 'answer' elements and their respective 'stance' elements. You must provide one overall prediction for each claim in the 'pred_label' attribute in the JSON.

**EXAMPLES WITH DETAILED REASONING:**

**Example 1:**
Claim 1: Wearing face masks will stop the spread of COVID-19.
Question 1: Does a face mask prevent the spread of COVID-19?
Evidence 1: Cloth face coverings, even homemade ones, are effective in reducing the spread of COVID-19, according to Oxford's Leverhulme Centre.
Stance 1: Positive

**Chain of Thought Analysis 1:**
1. **CLAIM ANALYSIS**: The claim asserts that face masks will stop COVID-19 spread.
2. **EVIDENCE EXAMINATION**: The evidence cites a credible Oxford study showing masks are effective in reducing spread. The stance is positive, indicating the evidence supports the claim.
3. **INDIVIDUAL EVIDENCE ASSESSMENT**: The evidence strongly supports the claim with academic research, and the positive stance confirms this alignment.
4. **PATTERN ANALYSIS**: Single piece of evidence from reliable source with positive stance consistently supports the claim.
5. **SYNTHESIS**: The Oxford study provides strong scientific backing for mask effectiveness, reinforced by the positive stance.
6. **FINAL CLASSIFICATION**: The evidence reliably supports the claim.

Veracity Prediction 1: Support
Reasoning 1: The evidence from Oxford's Leverhulme Centre provides credible scientific support for mask effectiveness, with a positive stance confirming the evidence aligns with the claim.

**Example 2:**
Claim 2: Trump Administration claimed Billie Eilish is destroying the country.
Question 2: Has the Trump administration made that claim?
Evidence 2: A Washington Post article wrongly stated this; no official documents confirm the claim.
Stance 2: Negative

**Chain of Thought Analysis 2:**
1. **CLAIM ANALYSIS**: The claim asserts the Trump administration made specific statements about Billie Eilish.
2. **EVIDENCE EXAMINATION**: The evidence shows a Washington Post correction indicating the claim was false. The stance is negative, indicating the evidence contradicts the claim.
3. **INDIVIDUAL EVIDENCE ASSESSMENT**: The evidence directly refutes the claim by showing it was based on incorrect reporting, and the negative stance confirms this contradiction.
4. **PATTERN ANALYSIS**: Single piece of evidence from reliable source with negative stance consistently refutes the claim.
5. **SYNTHESIS**: The correction from a credible news source demonstrates the claim was unfounded, supported by the negative stance.
6. **FINAL CLASSIFICATION**: The evidence reliably contradicts the claim.

Veracity Prediction 2: Refute
Reasoning 2: The evidence explicitly debunks the claim by showing it was based on incorrect reporting, with a negative stance confirming the evidence contradicts the claim.

**Example 3:**
Claim 3: It makes no sense for oil to be cheaper in Nigeria than in Saudi Arabia.
Question 3: Why do fuel prices differ by country?
Evidence 3: Fuel prices vary due to taxes, refining costs, and other components.
Stance 3: Neutral

**Chain of Thought Analysis 3:**
1. **CLAIM ANALYSIS**: The claim makes a judgment about the logic of oil pricing differences between two specific countries.
2. **EVIDENCE EXAMINATION**: The evidence provides general information about fuel price components but doesn't address the specific comparison. The stance is neutral, indicating the evidence neither supports nor refutes the claim.
3. **INDIVIDUAL EVIDENCE ASSESSMENT**: The evidence is relevant but doesn't directly evaluate the Nigeria vs Saudi Arabia comparison, and the neutral stance confirms this lack of direct connection.
4. **PATTERN ANALYSIS**: Single piece of evidence provides background information but lacks specific analysis needed, with neutral stance indicating insufficient direction.
5. **SYNTHESIS**: General pricing factors don't address the specific claim about these two countries, confirmed by the neutral stance.
6. **FINAL CLASSIFICATION**: Insufficient evidence to determine the validity of the specific comparison.

Veracity Prediction 3: Not Enough Evidence
Reasoning 3: The evidence provides general background on fuel pricing factors but does not specifically evaluate the comparison between Nigeria and Saudi Arabia, with a neutral stance confirming the lack of direct support or refutation.

**REASONING PROCESS - Follow these steps systematically:**

1. **CLAIM ANALYSIS**: First, identify the key assertions and components of the claim that need to be verified.

2. **EVIDENCE EXAMINATION**: For each piece of evidence provided:
   - Summarize what the evidence states
   - Assess the quality and reliability of the evidence
   - Determine how directly it relates to the claim
   - Consider how the stance (Positive/Negative/Neutral) aligns with your assessment

3. **INDIVIDUAL EVIDENCE ASSESSMENT**: For each evidence piece, determine if it:
   - Supports the claim (and to what degree)
   - Refutes the claim (and to what degree)
   - Is neutral/irrelevant to the claim
   - Contains conflicting information
   - Verify consistency between evidence content and assigned stance

4. **PATTERN ANALYSIS**: Look across all evidence pieces for:
   - Consistent patterns of support or refutation
   - Contradictions between different evidence sources
   - Gaps in information needed to verify the claim
   - Signs of selective evidence presentation (cherry-picking)
   - Patterns in stance assignments and their consistency with evidence content

5. **SYNTHESIS**: Combine your analysis of all evidence to determine the overall relationship between the evidence set and the claim, considering both content and stance information.

6. **FINAL CLASSIFICATION**: Based on your analysis, classify as:
   - **Supported**: Evidence consistently and reliably supports the claim
   - **Refuted**: Evidence consistently and reliably contradicts the claim
   - **Not Enough Evidence**: Insufficient reliable evidence to make a determination
   - **Conflicting/Cherrypicking**: Evidence presents contradictory information or appears selectively chosen

**CLAIM TO EVALUATE:**
{claim}

**EVIDENCE TO ANALYZE:**"""
    
    # Add questions, evidences, and stances
    for idx, (question, evidence, stance) in enumerate(zip(questions, evidences, stances)):
        prompt += f"""
Question {idx+1}: {question}
Evidence {idx+1}: {evidence}
Stance {idx+1}: {stance}"""
    
    prompt += f"""

**INSTRUCTIONS FOR YOUR RESPONSE:**
Please work through each step of the reasoning process outlined above. Show your thinking clearly for each step before providing your final answer.

Taking into account all the 'answer' elements in 'evidence' and their respective stances, you must predict the veracity towards the claim. The veracity prediction must be one overall prediction for the whole 'evidence' attribute towards the claim, so there must only be one prediction label and reasoning per claim.

Provide your response as a JSON with the following structure:
{{
    "pred_label": " 'Supported' / 'Refuted' / 'Not Enough Evidence' / 'Conflicting/Cherrypicking' ",
    "reasoning": "Your comprehensive reasoning for the final classification"
}}
Your answer must be in {language_instruction}."""

    return prompt.strip()



