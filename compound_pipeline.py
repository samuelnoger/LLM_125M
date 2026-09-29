import spacy
from LLM.model import load_extractor

class CompoundOrchestrator:
    def __init__(self, extractor_path="checkpoints/extractor_66M", min_confidence: float = 0.25):
        self.extractor = load_extractor(extractor_path)
        self.nlp = spacy.load("en_core_web_sm")
        self.min_confidence = min_confidence

    def process_query(self, user_query: str, raw_context: str, conversation_memory: list = None) -> tuple[str, str]:
        query_lower = user_query.lower()

        # 1. PRONOUN RESOLUTION (Query Expansion)
        pronouns = {"it", "its", "they", "their", "this", "that", "he", "she", "his", "her"}
        has_pronoun = any(p in query_lower.split() for p in pronouns)
        
        eval_query = user_query
        if has_pronoun and conversation_memory and len(conversation_memory) > 0:
            last_user, _ = conversation_memory[-1]
            # Merge the previous question to resolve pronouns (e.g., "Who was Einstein? What did he do?")
            eval_query = f"{last_user} {user_query}"

        # If there is no system text, just return the rewritten query and empty context
        if not raw_context.strip():
            return eval_query, ""

        # 2. SENTENCE SEGMENTATION & EXTRACTION
        doc_sentences = [sent.text.strip() for sent in self.nlp(raw_context).sents if sent.text.strip()]
        
        scored_sentences = []
        for idx, sentence in enumerate(doc_sentences):
            span, score = self.extractor.extract_span(question=eval_query, context=sentence)
            if score >= self.min_confidence and span:
                scored_sentences.append((idx, sentence, score))

        # 3. CONTEXT BUILDING
        if scored_sentences:
            scored_sentences.sort(key=lambda x: x[0])
            structured_context = "Background Information:\n"
            for _, sentence, _ in scored_sentences:
                structured_context += f"- {sentence}\n"
            
            # Return BOTH the rewritten query and the context
            return eval_query, structured_context

        return eval_query, ""