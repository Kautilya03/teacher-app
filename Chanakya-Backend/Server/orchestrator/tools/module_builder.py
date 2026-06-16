"""
Module Builder Tool
===================

Handles chapter selection, lesson generation, and assignment generation
directly through the chat interface. Supports iterative slide and assignment edits.
"""

import json
import logging
from typing import Dict, Any, Optional
from google import genai
from google.genai import types

from module.services.topic_selector import TopicSelectorService
from module.services.lesson_storage import LessonStorageService
from module.generators.lesson_generator import LessonGenerator
from module.generators.assignment_generator import AssignmentGenerator
from services.ragflow_v2 import ragflow_service

logger = logging.getLogger(__name__)


class ModuleBuilderTool:
    """
    Tool for building and refining lesson plans and assignments in chat.
    """
    
    name: str = "module_builder"
    description: str = "Generate textbook-aligned lesson plans, slides, and worksheets dynamically via chat."
    
    def __init__(self, api_key: str):
        """Initialize the module builder tool and its dependent services."""
        try:
            self.api_key = api_key
            self.client = genai.Client(api_key=api_key)
            self.model_name = "models/gemini-2.5-flash"
            
            # Initialize internal builders and database services
            self.topic_selector = TopicSelectorService()
            self.lesson_gen = LessonGenerator(api_key=api_key)
            self.assignment_gen = AssignmentGenerator(api_key=api_key)
            self.storage = LessonStorageService()
        except Exception as e:
            import traceback
            print("="*80)
            print("❌ ModuleBuilderTool.__init__ failed:")
            traceback.print_exc()
            print("="*80)
            raise e

    async def _parse_intent(self, query: str) -> Dict[str, Any]:
        """Parse class and subject from user's initiation request using Gemini."""
        prompt = f"""Analyze this user request for a lesson module and extract the target Class level, Subject area, and Chapter/Topic if specified.
        
User Request: "{query}"

Standardize class names as: Class_1, Class_2, ..., Class_12.
Standardize subject names (e.g. Mathematics, Science, Social Science, Geography).

Return a JSON object with:
{{
    "class_name": "Class_7" or null,
    "subject": "Geography" or "Science" or null,
    "topic": "Water" or "Photosynthesis" or null
}}

JSON:"""
        try:
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json"
                )
            )
            return json.loads(response.text.strip())
        except Exception as e:
            logger.warning(f"Error parsing module builder intent: {e}")
            return {}

    async def _handle_refinement(self, query: str, lesson_id: str, context: dict) -> dict:
        """Apply slide or question edits requested by the user to the existing lesson draft."""
        try:
            lesson = await self.storage.get_lesson(lesson_id)
            if not lesson:
                return {
                    "response": f"I couldn't find a lesson with ID '{lesson_id}' to edit.",
                    "status": "error"
                }
            
            assignment = await self.storage.get_assignment_for_lesson(lesson_id)
            
            # Use Gemini to edit slides or assignment based on user request
            prompt = f"""You are an editor for an interactive educational module.
The user wants to refine/modify the lesson or the assignment.

CURRENT LESSON STATE (JSON):
{json.dumps(lesson.model_dump() if hasattr(lesson, 'model_dump') else lesson.dict(), indent=2)}

CURRENT ASSIGNMENT STATE (JSON):
{json.dumps(assignment.model_dump() if hasattr(assignment, 'model_dump') else assignment.dict(), indent=2) if assignment else "None"}

USER REFINEMENT REQUEST:
"{query}"

Apply the edits. You must keep all unchanged parts intact. You must output the modified lesson and/or assignment conforming EXACTLY to the structure of the input JSON.

If the user request is related to changing slides, modify only the lesson.
If the user request is related to changing questions/assignment, modify only the assignment.
If both need updating, modify both.
For the elements you did NOT modify, return them as they were.

Make sure total_marks in assignment matches the sum of the marks of all questions.
Each MCQ question must have exactly 4 options with exactly one correct option.
Each slide must be structured exactly like the current state slides.

Return a JSON object containing:
{{
    "updated_element": "lesson" or "assignment" or "both",
    "lesson": <updated lesson object matching the input schema>,
    "assignment": <updated assignment object matching the input schema if present, else null>,
    "explanation": "Brief explanation of what was updated"
}}

Respond with valid JSON only. Do not wrap in markdown tags or anything else."""

            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=[types.Content(role="user", parts=[types.Part(text=prompt)])],
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json"
                )
            )
            
            result = json.loads(response.text.strip())
            explanation = result.get("explanation", "Updated the module successfully.")
            
            updated_lesson_dict = result.get("lesson")
            updated_assignment_dict = result.get("assignment")
            
            from module.models.schemas import Lesson, Assignment
            
            if updated_lesson_dict:
                updated_lesson_dict["id"] = lesson_id
                if "created_at" not in updated_lesson_dict:
                    updated_lesson_dict["created_at"] = lesson.created_at.isoformat()
                lesson = Lesson(**updated_lesson_dict)
                
            if updated_assignment_dict:
                updated_assignment_dict["lesson_id"] = lesson_id
                if "id" not in updated_assignment_dict:
                    updated_assignment_dict["id"] = assignment.id if assignment else None
                if "created_at" not in updated_assignment_dict:
                    updated_assignment_dict["created_at"] = assignment.created_at.isoformat() if assignment else None
                
                # Re-calculate total marks
                questions = updated_assignment_dict.get("questions", [])
                total_marks = sum(q.get("marks", 0) for q in questions)
                updated_assignment_dict["total_marks"] = total_marks
                
                assignment = Assignment(**updated_assignment_dict)
            
            # Save updates
            await self.storage.save_lesson(lesson, assignment)
            
            return {
                "response": f"Refined successfully: {explanation}. The preview has been updated.",
                "lesson_id": lesson_id,
                "lesson": lesson.model_dump() if hasattr(lesson, 'model_dump') else lesson.dict(),
                "assignment": assignment.model_dump() if hasattr(assignment, 'model_dump') else assignment.dict() if assignment else None,
                "status": "preview_module"
            }
            
        except Exception as e:
            logger.error(f"Error refining module: {e}", exc_info=True)
            return {
                "response": f"Failed to apply refinements: {str(e)}",
                "status": "error"
            }

    async def run(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute the module builder tool.
        """
        context = context or {}
        lesson_id = context.get("lesson_id")
        
        # 1. Edit an existing module if lesson_id is present
        if lesson_id:
            return await self._handle_refinement(query, lesson_id, context)
            
        # 2. Retrieve metadata context (class, subject, topic)
        class_name = context.get("class_name")
        subject = context.get("subject")
        topic = context.get("topic")
        
        if not class_name or not subject:
            parsed = await self._parse_intent(query)
            class_name = parsed.get("class_name") or class_name
            subject = parsed.get("subject") or subject
            topic = parsed.get("topic") or topic
            
        if not class_name or not subject:
            return {
                "response": "To help you build a lesson plan, please specify the class (e.g. Class 7) and subject (e.g. Geography) in the chat.",
                "status": "request_metadata"
            }
            
        # Standardize class and subject formats
        class_name_std = class_name.replace(" ", "_").strip()
        subject_std = subject.strip().capitalize()
        
        # 3. If no topic is selected yet, offer chapter choices
        if not topic:
            try:
                topics = await self.topic_selector.get_topics_for_subject(class_name_std, subject_std)
                if not topics:
                    return {
                        "response": f"I couldn't find any NCERT chapters for {class_name} {subject}.",
                        "status": "error"
                    }
                    
                topic_list = [{"topic_name": t.topic_name, "chapter_number": t.chapter_number} for t in topics]
                return {
                    "response": f"I found the following chapters in the NCERT book for **{class_name} {subject_std}**. Please choose a chapter to build the module for:",
                    "topics": topic_list,
                    "class_name": class_name_std,
                    "subject": subject_std,
                    "status": "select_topic"
                }
            except Exception as e:
                logger.error(f"Error fetching topics: {e}", exc_info=True)
                return {
                    "response": f"Failed to retrieve chapters: {str(e)}",
                    "status": "error"
                }
                
        # 4. Generate the draft lesson and assignment
        try:
            logger.info(f"Generating module for {class_name_std}/{subject_std}/{topic}")
            language = context.get("language", "English")
            board = context.get("board", "NCERT")
            
            # Fetch content from RAGFlow
            textbook_content, retrieval_meta = await ragflow_service.retrieve_textbook_content(
                class_name_std,
                subject_std,
                topic,
                language=language,
                board=board,
            )
            
            if not textbook_content:
                from config import settings
                return {
                    "response": f"⚠️ **RAGFlow Retrieval Failed**: No textbook chunks could be retrieved from RAGFlow for chapter **{topic}** (Class: {class_name_std}, Subject: {subject_std}).\n\nPlease ensure the NCERT textbook PDF is uploaded to your RAGFlow knowledge base (Dataset ID: `{settings.RAGFLOW_DATASET_ID}`) and fully indexed.",
                    "status": "error"
                }
                
            # Draft lesson slides
            lesson, validation_report = await self.lesson_gen.generate_lesson(
                textbook_content=textbook_content,
                class_name=class_name_std,
                subject=subject_std,
                topic=topic
            )
            
            # Draft assignment questions
            assignment = await self.assignment_gen.generate_assignment(
                lesson=lesson,
                textbook_content=textbook_content
            )
            
            # Save drafts to DB
            lesson_id = await self.storage.save_lesson(lesson, assignment)
            lesson.id = lesson_id
            assignment.lesson_id = lesson_id
            
            lesson_dict = lesson.model_dump() if hasattr(lesson, 'model_dump') else lesson.dict()
            assignment_dict = assignment.model_dump() if hasattr(assignment, 'model_dump') else assignment.dict()
            validation_dict = validation_report.model_dump() if hasattr(validation_report, 'model_dump') else validation_report.dict()
            
            return {
                "response": f"I have generated a draft module for you on **'{topic}'** ({class_name} {subject_std})! You can preview the interactive slides and assignment questions in the Artifact pane on the right.",
                "lesson_id": lesson_id,
                "class_name": class_name_std,
                "subject": subject_std,
                "topic": topic,
                "lesson": lesson_dict,
                "assignment": assignment_dict,
                "validation_report": validation_dict,
                "status": "preview_module"
            }
            
        except Exception as e:
            logger.error(f"Error generating module: {e}", exc_info=True)
            return {
                "response": f"Sorry, I encountered an error while generating the lesson plan: {str(e)}",
                "status": "error"
            }
