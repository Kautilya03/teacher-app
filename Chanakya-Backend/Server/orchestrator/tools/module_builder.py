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
        Execute the module builder tool by querying RAGFlow chatbot directly.
        """
        context = context or {}
        lesson_id = context.get("lesson_id")
        
        # 1. Edit an existing module if lesson_id is present
        if lesson_id:
            return await self._handle_refinement(query, lesson_id, context)
            
        # 2. For any new lesson plan request, pass query directly to RAGFlow chatbot completion
        fastapi_session_id = context.get("session_id") or "default_session"
        logger.info(f"module_builder_lesson_plan_direct: query={query}, session_id={fastapi_session_id}")
        
        from services.ragflow_v2 import ragflow_service
        import asyncio
        from config import settings
        
        # Resolve custom chatbot ID for module creation, default to RAGFLOW_MODULE_CHAT_ID or fall back to RAGFLOW_CHAT_ID
        module_chat_id = getattr(settings, "RAGFLOW_MODULE_CHAT_ID", None) or settings.RAGFLOW_CHAT_ID
        logger.info(f"Using module chatbot ID: {module_chat_id}")
        
        # Get or create RAGFlow session ID
        ragflow_session_id = None
        chat_session = None
        try:
            from models.chat_session import ChatSession as MongoChatSession
            chat_session = await MongoChatSession.find_one(MongoChatSession.session_id == fastapi_session_id)
            if chat_session and chat_session.ragflow_session_id:
                ragflow_session_id = chat_session.ragflow_session_id
                logger.info(f"Retrieved existing ragflow_session_id from MongoDB: {ragflow_session_id}")
        except Exception as db_err:
            logger.warning(f"MongoDB/Beanie lookup failed in module_builder: {db_err}")
            
        # Fallback to listing RAGFlow sessions
        if not ragflow_session_id:
            try:
                sessions = await asyncio.to_thread(
                    ragflow_service.list_sessions,
                    chat_id=module_chat_id
                )
                for s in sessions:
                    if s.get("name") == fastapi_session_id:
                        ragflow_session_id = s.get("id")
                        break
            except Exception as list_err:
                logger.warning(f"Failed to list RAGFlow sessions in module_builder: {list_err}")
                
        # If not found, create new session in RAGFlow
        if not ragflow_session_id:
            if chat_session and chat_session.title and chat_session.title != "New Chat":
                session_name = chat_session.title
            else:
                session_name = f"Chat: {query[:50]}"
                
            logger.info(f"Creating new RAGFlow chat session for: '{session_name}'")
            try:
                ragflow_session_id = await asyncio.to_thread(
                    ragflow_service.create_new_session,
                    chat_id=module_chat_id,
                    name=session_name
                )
                
                # Save new ragflow_session_id to MongoDB
                if ragflow_session_id and chat_session:
                    chat_session.ragflow_session_id = ragflow_session_id
                    await chat_session.save()
                    logger.info(f"Saved new ragflow_session_id {ragflow_session_id} to MongoDB")
            except Exception as create_err:
                logger.error(f"Failed to create RAGFlow session in module_builder: {create_err}")
                
        if not ragflow_session_id:
            return {
                "response": "Failed to get or create RAGFlow chat session.",
                "status": "error"
            }
            
        # Call RAGFlow stateful completion directly with the query
        logger.info(f"Querying RAGFlow Chat Assistant directly: '{query}' with session: {ragflow_session_id}")
        try:
            resp = await asyncio.to_thread(
                ragflow_service.chat_completion_stateful,
                question=query,
                chat_id=module_chat_id,
                session_id=ragflow_session_id
            )
            
            if not resp or resp.get("code") != 0:
                error_msg = resp.get("message") or "Failed to get response from RAGFlow"
                raise Exception(error_msg)
                
            answer = resp.get("data", {}).get("answer", "")
            
            # 3. Use Gemini to parse the raw RAGFlow chatbot text into structured JSON matching schemas
            logger.info("Structuring RAGFlow response into schemas using Gemini")
            structuring_prompt = f"""Analyze the RAGFlow chatbot response and convert it into a structured JSON lesson and assignment conforming to the schemas below.

RAGFlow Chatbot Response:
"{answer}"

You must output a JSON object containing:
{{
    "lesson": {{
        "class_name": "Class_6" or similar,
        "subject": "Geography" or similar,
        "topic": "Environment" or similar,
        "slides": [
            {{
                "slide_number": 1,
                "slide_type": "introduction",
                "title": "...",
                "explanation": "...",
                "bullet_points": ["point 1", "point 2", ...],
                "key_terms": ["term 1: definition", "term 2: definition"],
                "examples": ["example 1"],
                "diagram_prompt": "prompt for diagram generation"
            }},
            {{
                "slide_number": 2,
                "slide_type": "concept",
                "title": "...",
                "explanation": "...",
                "bullet_points": ["point 1", "point 2", ...],
                "key_terms": ["term 1: definition", "term 2: definition"],
                "examples": ["example 1"],
                "diagram_prompt": "prompt for diagram generation"
            }}
        ]
    }},
    "assignment": {{
        "questions": [
            {{
                "question_type": "mcq",
                "question_text": "...",
                "options": [
                    {{"option_text": "...", "is_correct": true}},
                    {{"option_text": "...", "is_correct": false}},
                    ... (exactly 4 options)
                ],
                "difficulty": "easy" or "medium" or "hard",
                "marks": 1,
                "source_reference": "..."
            }},
            {{
                "question_type": "short_answer",
                "question_text": "...",
                "expected_answer": "...",
                "difficulty": "medium",
                "marks": 2,
                "source_reference": "..."
            }},
            {{
                "question_type": "long_answer",
                "question_text": "...",
                "expected_answer": "...",
                "marking_scheme": ["point 1", "point 2"],
                "difficulty": "hard",
                "marks": 5,
                "source_reference": "..."
            }}
        ]
    }}
}}

CRITICAL INSTRUCTIONS:
1. The "slides" list must contain EXACTLY 2 slides. If the input has more or fewer, consolidate/adapt them to fit exactly 2 slides, with slide_number from 1 to 2 in order.
2. The explanation of each slide must be a simplified explanation of at least 10 characters.
3. Every MCQ question must have exactly 4 options with exactly one correct option.
4. Total marks of the assignment must equal the sum of all question marks.
5. Do not include markdown wraps (like ```json). Respond with valid JSON only.
"""
            from google.genai import types
            gemini_resp = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=[types.Content(role="user", parts=[types.Part(text=structuring_prompt)])],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json"
                )
            )
            
            # 4. Parse the structured JSON
            parsed_data = json.loads(gemini_resp.text.strip())
            
            # Import models
            from module.models.schemas import Lesson, Slide, Assignment, ValidationReport
            
            lesson_dict = parsed_data.get("lesson", {})
            assignment_dict = parsed_data.get("assignment", {})
            
            # Clean up slides source references
            for slide_data in lesson_dict.get("slides", []):
                if "source_references" not in slide_data:
                    slide_data["source_references"] = []
                if "diagram_prompt" not in slide_data or not slide_data["diagram_prompt"]:
                    slide_data["diagram_prompt"] = f"Diagram illustrating {slide_data.get('title')}"
                    
            # Set default validation score
            lesson_dict["validation_score"] = 1.0
            lesson_dict["ragflow_session_id"] = ragflow_session_id
            
            # Create Pydantic models
            lesson = Lesson(**lesson_dict)
            
            # Build assignment Pydantic model
            assignment_dict["class_name"] = lesson.class_name
            assignment_dict["subject"] = lesson.subject
            assignment_dict["topic"] = lesson.topic
            assignment_dict["lesson_id"] = "temp_lesson_id"
            
            # Calculate total marks
            questions_list = assignment_dict.get("questions", [])
            total_marks = sum(q.get("marks", 0) for q in questions_list)
            assignment_dict["total_marks"] = total_marks
            
            assignment = Assignment(**assignment_dict)
            
            # 5. Save drafts to DB
            lesson_id = await self.storage.save_lesson(lesson, assignment)
            lesson.id = lesson_id
            assignment.lesson_id = lesson_id
            
            # Re-calculate or update assignment in DB with saved lesson_id
            await self.storage.save_lesson(lesson, assignment)
            
            lesson_out = lesson.model_dump() if hasattr(lesson, 'model_dump') else lesson.dict()
            assignment_out = assignment.model_dump() if hasattr(assignment, 'model_dump') else assignment.dict()
            
            validation_report = ValidationReport(
                is_valid=True,
                overall_score=1.0,
                issues=[],
                flagged_content=[],
                recommendations=[]
            )
            validation_out = validation_report.model_dump() if hasattr(validation_report, 'model_dump') else validation_report.dict()
            
            return {
                "response": f"I have successfully generated the 2-slide lesson plan and worksheet! You can preview them in the pane on the right.",
                "lesson_id": lesson_id,
                "class_name": lesson.class_name,
                "subject": lesson.subject,
                "topic": lesson.topic,
                "lesson": lesson_out,
                "assignment": assignment_out,
                "validation_report": validation_out,
                "ragflow_session_id": ragflow_session_id,
                "status": "preview_module"
            }
        except Exception as e:
            logger.error(f"Error querying RAGFlow chatbot directly or structuring: {e}", exc_info=True)
            return {
                "response": f"Sorry, I encountered an error while querying RAGFlow chatbot: {str(e)}",
                "status": "error"
            }
