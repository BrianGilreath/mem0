import logging
import json
from typing import List

from app.utils.prompts import MEMORY_CATEGORIZATION_PROMPT
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15))
def get_categories_for_memory(memory: str) -> List[str]:
    try:
        from app.utils.memory import get_memory_client
        messages = [
            {"role": "system", "content": MEMORY_CATEGORIZATION_PROMPT},
            {"role": "user", "content": memory}
        ]

        client = get_memory_client()
        if client is None or not hasattr(client, "llm"):
            raise RuntimeError("Memory client or its LLM is not available.")

        # Use the LLM's generate_response method
        response = client.llm.generate_response(
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0
        )

        parsed = json.loads(response)
        return [cat.strip().lower() for cat in parsed.get("categories", [])]

    except Exception as e:
        logging.error(f"[ERROR] Failed to get categories: {e}")
        try:
            logging.debug(f"[DEBUG] Raw response: {completion.choices[0].message.content}")
        except Exception as debug_e:
            logging.debug(f"[DEBUG] Could not extract raw response: {debug_e}")
        raise
