
# from langchain_community.llms import GPT4All
# from langchain.prompts import PromptTemplate
# from langchain_core.runnables import RunnableSequence
# import os
# import json
# import re
# import logging

# # Suppress CUDA errors
# os.environ["GPT4ALL_NO_CUDA"] = "1"

# # Configure logging to file
# logging.basicConfig(filename='gpt4all_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')
# logger = logging.getLogger()

# base_dir = os.path.dirname(os.path.abspath(__file__))
# model_file = "Phi-3-mini-4k-instruct.Q4_0.gguf"
# model_path = os.path.join(base_dir, "../gptmodel", model_file)

# # Verify model file exists
# if not os.path.exists(model_path):
#     logger.error(f"Model file not found: {model_path}")
#     print(json.dumps({"job_title": "Not detected", "skills": []}, indent=2))
#     exit(1)

# # Initialize GPT4All
# try:
#     llm = GPT4All(
#         model=model_path,
#         verbose=True,
#         backend="cpu",
#         temp=0.5,
#         max_tokens=100,  # Further reduce tokens
#         n_threads=1,  # Minimize threads
#         n_batch=1
#     )
#     logger.info("Model loaded successfully")
# except Exception as e:
#     logger.error(f"Failed to load model: {str(e)}")
#     print(json.dumps({"job_title": "Not detected", "skills": []}, indent=2))
#     exit(1)

# # Enhanced prompt
# prompt_template = PromptTemplate(
#     input_variables=["cv_text"],
#     template="""
# You are an AI that analyzes CV/resume text to extract the job title and skills. Return ONLY valid JSON with no additional text, prose, explanations, or tokens (e.g., <|endoftext|>).

# - Identify the most likely job title from qualifications, experience, certifications, or explicit mentions (e.g., "job title:").
# - Extract ALL technical skills mentioned, including programming languages, tools, frameworks, and methodologies (e.g., "responsive design").
# - If no job title or skills are detected, return empty strings and arrays.
# - Do not generate unrelated content or continue beyond the JSON output.

# Example output:
# {{
#   "job_title": "data analyst",
#   "skills": ["python", "sql", "excel"]
# }}

# CV:
# {cv_text}
# """
# )

# # Create RunnableSequence
# llm_chain = RunnableSequence(prompt_template | llm)

# def extract_json_from_response(response: str) -> dict:
#     """Extract valid JSON from model response."""
#     response = response.replace("<|endoftext|>", "").strip()
#     json_match = re.search(r'\{[\s\S]*\}', response)
#     if json_match:
#         try:
#             return json.loads(json_match.group(0))
#         except json.JSONDecodeError:
#             logger.error(f"Invalid JSON in response: {response}")
#     else:
#         logger.error(f"No JSON found in response: {response}")
#     return {"job_title": "Not detected", "skills": []}

# def extract_job_title_and_skills_with_llm(cv_text: str):
#     try:
#         response = llm_chain.invoke({"cv_text": cv_text[:1000]})  # Further reduce input size
#         logger.info(f"Raw response: {response}")
#         return extract_json_from_response(response)
#     except Exception as e:
#         logger.error(f"LLM extraction failed: {str(e)}")
#         return {"job_title": "Not detected", "skills": []}

# if __name__ == "__main__":
#     cv_text = """
# Contact Information: tbi@example.com | +1-1234567890

# Summary of Qualifications:
# Highly skilled in front-end development with expertise in HTML5, CSS3, JavaScript (including React), and responsive design. Proficient in back-end technologies such as Node.js, Express, MongoDB, and PostgreSQL.
# job title: frontend development
# Technical Skills:
# * Front-end: HTML5, CSS3, JavaScript
# * Back-end: Node.js, Express, MongoDB, PostgreSQL

# Work Experience:
# Tobi worked at ABC Company from 2018 to 2020.
# During this time, he developed multiple web applications using React, Node.js, and MongoDB. He also implemented responsive design for several projects.

# Education:
# Bachelor of Science in Computer Science
# University of XYZ

# Certifications:
# * Certified Web Developer (CWD)
# * Certified Front-end Developer (CFD)

# Personal Projects:
# Tobi has worked on various personal projects that demonstrate his skills in front-end development, including a web application built using React and Node.js. He also created a responsive design for an e-commerce website.
# """
#     result = extract_job_title_and_skills_with_llm(cv_text)
#     print(json.dumps(result, indent=2))


# from langchain_community.llms import GPT4All
# from langchain.prompts import PromptTemplate
# from langchain_core.runnables import RunnableSequence
# import os
# import json
# import re
# import logging
# import time

# # Suppress CUDA errors
# os.environ["GPT4ALL_NO_CUDA"] = "1"

# # Configure logging
# logging.basicConfig(filename='gpt4all_log.txt', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger()

# base_dir = os.path.dirname(os.path.abspath(__file__))
# model_file = "Phi-3-mini-4k-instruct.Q4_0.gguf"
# model_path = os.path.join(base_dir, "../gptmodel", model_file)

# # Verify model file
# if not os.path.exists(model_path):
#     logger.error(f"Model file not found: {model_path}")
#     raise FileNotFoundError(f"Model file not found: {model_path}")

# # Initialize GPT4All
# try:
#     llm = GPT4All(
#         model=model_path,
#         verbose=True,
#         backend="cpu",
#         temp=0.5,
#         max_tokens=100,
#         n_threads=1,
#         n_batch=1
#     )
#     logger.info("Model loaded successfully")
# except Exception as e:
#     logger.error(f"Failed to load model: {str(e)}")
#     raise RuntimeError(f"Failed to load model: {str(e)}")

# # Enhanced prompt
# prompt_template = PromptTemplate(
#     input_variables=["cv_text"],
#     template="""
# You are an AI that analyzes CV/resume text to extract the job title and skills. Return ONLY valid JSON with no additional text, prose, explanations, or tokens (e.g., <|endoftext|>).

# - Identify the most likely job title from qualifications, experience, certifications, or explicit mentions (e.g., "job title:").
# - Extract ALL technical skills mentioned, including programming languages, tools, frameworks, and methodologies (e.g., "responsive design").
# - If no job title or skills are detected, return empty strings and arrays.
# - Do not generate unrelated content or continue beyond the JSON output.

# Example output:
# {{
#   "job_title": "data analyst",
#   "skills": ["python", "sql", "excel"]
# }}

# CV:
# {cv_text}
# """
# )

# # Create RunnableSequence
# llm_chain = RunnableSequence(prompt_template | llm)

# def extract_json_from_response(response: str) -> dict:
#     response = response.replace("<|endoftext|>", "").strip()
#     json_match = re.search(r'\{[\s\S]*\}', response)
#     if json_match:
#         try:
#             return json.loads(json_match.group(0))
#         except json.JSONDecodeError:
#             logger.error(f"Invalid JSON in response: {response}")
#     else:
#         logger.error(f"No JSON found in response: {response}")
#     return {"job_title": "Not detected", "skills": []}

# def extract_job_title_and_skills_with_llm(cv_text: str):
#     start_time = time.time()
#     try:
#         cv_text = cv_text[:800]  # Further reduce input size
#         logger.debug(f"Processing CV text (length: {len(cv_text)})")
#         response = llm_chain.invoke({"cv_text": cv_text})
#         logger.info(f"Raw response: {response}")
#         result = extract_json_from_response(response)
#         logger.info(f"Extraction took {time.time() - start_time:.2f} seconds")
#         return result
#     except Exception as e:
#         logger.error(f"LLM extraction failed: {str(e)}")
#         return {"job_title": "Not detected", "skills": []}


# import os
# import json
# import re
# import logging
# import time
# import traceback
# from langchain_community.llms import GPT4All
# from langchain.prompts import PromptTemplate
# from langchain_core.runnables import RunnableSequence
# from llama_cpp import Llama

# # Suppress CUDA errors
# os.environ["GPT4ALL_NO_CUDA"] = "1"

# # Configure logging
# logging.basicConfig(filename='gpt4all_log.txt', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger()

# base_dir = os.path.dirname(os.path.abspath(__file__))


# model_file = "Phi-3-mini-4k-instruct.Q4_0.gguf"
# model_path = os.path.abspath(os.path.join(base_dir, "../gptmodel", model_file))

# llm = Llama(model_path=model_path)
# # Verify model file
# if not os.path.exists(model_path):
#     logger.error(f"Model file not found: {model_path}")
#     raise FileNotFoundError(f"Model file not found: {model_path}")

# # Initialize GPT4All
# try:
#     llm = GPT4All(
#     model=model_path,
#     backend="llama",         # or "cpu" if "llama" fails — try both
#     verbose=True,
#     temp=0.5,
#     max_tokens=256,          
#     n_threads=8,             
#     n_batch=8                
# )
#     # llm = GPT4All(
#     #     model=model_path,
#     #     verbose=True,
#     #     backend="cpu",
#     #     temp=0.5,
#     #     max_tokens=100,
#     #     n_threads=1,
#     #     n_batch=1
#     # )
#     logger.info("Model loaded successfully")
# except Exception as e:
#     logger.error(f"Failed to load model: {str(e)}")
#     raise RuntimeError(f"Failed to load model: {str(e)}")

# # Enhanced prompt
# prompt_template = PromptTemplate(
#     input_variables=["cv_text"],
#     template="""
# You are an AI model that extracts structured information from resumes.

# Given the following CV content, do the following:
# - Extract the most likely job title based on the person's qualifications, experience, and keywords.
# - Extract all technical skills: programming languages, tools, libraries, frameworks, and technical concepts.
# - Return only valid JSON in this format:

# {{
#   "job_title": "job title here",
#   "skills": ["skill1", "skill2", "skill3"]
# }}

# Rules:
# - Do not include explanation, prose, or extra text.
# - Do not add tokens like <|endoftext|> or comments.
# - If no job title or skills found, return: {{"job_title": "", "skills": []}}

# CV:
# {cv_text}
# """
# )

# # Create RunnableSequence
# llm_chain = RunnableSequence(prompt_template | llm)

# def extract_json_from_response(response: str) -> dict:
#     response = response.replace("<|endoftext|>", "").strip()
#     json_match = re.search(r'\{[\s\S]*\}', response)
#     if json_match:
#         try:
#             parsed = json.loads(json_match.group(0))
#             if not isinstance(parsed, dict):
#                 logger.error(f"Parsed JSON is not a dict: {parsed}")
#                 raise ValueError("Parsed JSON is not a dict")
#             return parsed
#         except json.JSONDecodeError:
#             logger.error(f"Invalid JSON in response: {response}")
#             raise
#     else:
#         logger.error(f"No JSON found in response: {response}")
#         raise ValueError("No JSON found in LLM response")

# def extract_job_title_and_skills_with_llm(cv_text: str):
#     start_time = time.time()
#     try:
#         cv_text = cv_text[:800]  # Limit input size for speed
#         logger.debug(f"Processing CV text (length: {len(cv_text)})")
#         response = llm_chain.invoke({"cv_text": cv_text})
#         logger.info(f"Raw response: {response}")

#         result = extract_json_from_response(response)
        
#         job_title = result.get("job_title", "").strip()
#         skills = result.get("skills", [])

#         # Validate output
#         if not job_title or not isinstance(skills, list):
#             raise ValueError("Invalid extraction result: missing or malformed job_title or skills")

#         logger.info(f"Extraction took {time.time() - start_time:.2f} seconds")
#         return {
#             "job_title": job_title,
#             "skills": skills
#         }
#     except Exception as e:
#         logger.error("LLM extraction failed:")
#         logger.error(traceback.format_exc())
#         # Raise the error so FastAPI can handle it and return 500 with a message
#         raise RuntimeError(f"LLM extraction failed: {str(e)}")


# import os
# import json
# import re
# import logging
# import time
# import traceback
# from langchain_community.llms import GPT4All
# from langchain.prompts import PromptTemplate
# from langchain_core.runnables import RunnableSequence

# # Suppress CUDA errors
# os.environ["GPT4ALL_NO_CUDA"] = "1"

# # Logging config
# logging.basicConfig(filename='gpt4all_log.txt', level=logging.DEBUG,
#                     format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger()

# # Model path
# base_dir = os.path.dirname(os.path.abspath(__file__))
# model_file = "Phi-3-mini-4k-instruct.Q4_0.gguf"
# model_path = os.path.abspath(os.path.join(base_dir, "../gptmodel", model_file))

# if not os.path.exists(model_path):
#     logger.error(f"Model file not found: {model_path}")
#     raise FileNotFoundError(f"Model file not found: {model_path}")

# # Load GPT4All model
# try:
#     llm = GPT4All(
#         model=model_path,
#         backend="llama",
#         verbose=True,
#         temp=0.5,
#         max_tokens=256,
#         n_threads=8,
#         n_batch=8
#     )
#     logger.info("Model loaded successfully")
# except Exception as e:
#     logger.error(f"Failed to load model: {str(e)}")
#     raise RuntimeError(f"Failed to load model: {str(e)}")

# # Prompt
# prompt_template = PromptTemplate(
#     input_variables=["cv_text"],
#     template="""
# You are an AI model that extracts structured information from resumes.

# Given the following CV content, do the following:
# - Extract the most likely job title based on the person's qualifications, experience, and keywords.
# - Extract all technical skills: programming languages, tools, libraries, frameworks, and technical concepts.
# - Return only valid JSON in this format:

# {{
#   "job_title": "job title here",
#   "skills": ["skill1", "skill2", "skill3"]
# }}

# Rules:
# - Do not include explanation, prose, or extra text.
# - Do not add tokens like <|endoftext|> or comments.
# - If no job title or skills found, return: {{"job_title": "", "skills": []}}

# CV:
# {cv_text}
# """
# )

# llm_chain = RunnableSequence(prompt_template | llm)


# def extract_json_from_response(response: str) -> dict:
#     response = response.replace("<|endoftext|>", "").strip()
#     json_match = re.search(r'\{[\s\S]*?\}', response)
#     if json_match:
#         try:
#             parsed = json.loads(json_match.group(0))
#             if not isinstance(parsed, dict):
#                 raise ValueError("Parsed JSON is not a dict")
#             return parsed
#         except json.JSONDecodeError:
#             logger.error(f"Invalid JSON in response: {response}")
#             raise
#     raise ValueError("No valid JSON object found in response")


# def extract_job_title_and_skills_with_llm(cv_text: str):
#     start_time = time.time()
#     try:
#         cv_text = cv_text.strip()[:800]  # Truncate for speed
#         logger.debug(f"Invoking LLM on CV snippet (length: {len(cv_text)})")

#         max_attempts = 3
#         for attempt in range(max_attempts):
#             try:
#                 response = llm_chain.invoke({"cv_text": cv_text})
#                 logger.info(f"LLM raw response (attempt {attempt+1}): {response}")
#                 result = extract_json_from_response(response)
                
#                 job_title = result.get("job_title", "").strip()
#                 skills = result.get("skills", [])

#                 if isinstance(job_title, str) and isinstance(skills, list):
#                     logger.info(f"Extraction success in {time.time() - start_time:.2f}s")
#                     return {"job_title": job_title, "skills": skills}
#                 else:
#                     raise ValueError("Invalid format in parsed response")

#             except Exception as e:
#                 logger.warning(f"Attempt {attempt+1} failed: {str(e)}")
#                 if attempt == max_attempts - 1:
#                     raise

#     except Exception as e:
#         logger.error("LLM extraction failed:")
#         logger.error(traceback.format_exc())
#         raise RuntimeError(f"LLM extraction failed: {str(e)}")



import os
import json
import re
import logging
import time
import traceback
from llama_cpp import Llama

# Suppress CUDA errors
os.environ["GPT4ALL_NO_CUDA"] = "1"

# Configure logging
logging.basicConfig(filename='llama_log.txt', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

base_dir = os.path.dirname(os.path.abspath(__file__))
model_file = "Phi-3-mini-4k-instruct.Q4_0.gguf"
model_path = os.path.abspath(os.path.join(base_dir, "../gptmodel", model_file))

# Verify model file
if not os.path.exists(model_path):
    logger.error(f"Model file not found: {model_path}")
    raise FileNotFoundError(f"Model file not found: {model_path}")

# Initialize Llama model directly
try:
    llm = Llama(
        model_path=model_path,
        n_threads=8,
        n_batch=8,
        n_ctx=2048,
        verbose=True
    )
    logger.info("Llama model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load Llama model: {str(e)}")
    raise RuntimeError(f"Failed to load Llama model: {str(e)}")

# Prompt template
def build_prompt(cv_text):
    return f"""
You are an AI that extracts structured career data from resumes.

Your task is to:
1. Infer the most likely target job title based on the person's education, career goals, or intentions (even if not explicitly stated).
2. Extract all technical skills (languages, frameworks, platforms, tools, etc.).
3. Return only valid JSON in this format:

{{
  "job_title": "job title here",
  "skills": ["skill1", "skill2", "skill3"]
}}

Guidelines:
- Job title may be implied by phrases like “I want to become a…” or “Eager to work as…”.
- Do not include any explanation or text outside the JSON.
- Do not use two job titles like "ethical hacker/software engineer". Pick one like software engineer.
- Do not include words like intern or trainee.
- Do not use markdown, comments, or headings.

Here is the resume:
{cv_text}
"""

# Parse JSON from raw response
def extract_json_from_response(response: str) -> dict:
    response = response.replace("<|endoftext|>", "").strip()

    if not response:
        logger.error("❌ Model returned empty response.")
        return {"job_title": "", "skills": []}
    
    json_match = re.search(r'\{[\s\S]*\}', response)

    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            logger.error(f"⚠️ JSON parsing failed. Response:\n{response}")
    else:
        logger.error(f"❌ No JSON found in model response:\n{response}")
    
    return {"job_title": "", "skills": []}

    # if json_match:
    #     try:
    #         parsed = json.loads(json_match.group(0))
    #         if not isinstance(parsed, dict):
    #             logger.error(f"Parsed JSON is not a dict: {parsed}")
    #             raise ValueError("Parsed JSON is not a dict")
    #         return parsed
    #     except json.JSONDecodeError:
    #         logger.error(f"Invalid JSON in response: {response}")
    #         raise
    # else:
    #     logger.error(f"No JSON found in response: {response}")
    #     raise ValueError("No JSON found in LLM response")

# Main function to extract job title and skills
def extract_job_title_and_skills_with_llama(cv_text: str):
    start_time = time.time()
    try:
        cv_text = cv_text[:600]  # Limit input size for speed
        prompt = build_prompt(cv_text)

        logger.debug(f"Prompt:\n{prompt}")

        # Generate response from model
        response = llm(
            prompt,
            max_tokens=256,
            temperature=0.5,
            stop=["</s>", "```"]
        )["choices"][0]["text"]

        logger.info(f"Raw response: {response}")
        result = extract_json_from_response(response)

        job_title = result.get("job_title", "").strip()
        skills = result.get("skills", [])

        # if not job_title or not isinstance(skills, list):
        #     raise ValueError("Invalid extraction result: missing or malformed job_title or skills")

        if not isinstance(skills, list):
            skills = []

        if not job_title:
            logger.warning("No job title detected. Defaulting to 'Not detected'")
            job_title = "Not detected"


        logger.info(f"Extraction took {time.time() - start_time:.2f} seconds")
        return {
            "job_title": job_title,
            "skills": skills
        }

    except Exception as e:
        logger.error("Llama extraction failed:")
        logger.error(traceback.format_exc())
        raise RuntimeError(f"LLM extraction failed: {str(e)}")
