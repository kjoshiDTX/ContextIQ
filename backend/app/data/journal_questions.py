"""Static question bank for guided journal sessions.

Questions rotate through 6 categories. Each session presents 3 questions
drawn from different categories to ensure variety.
"""

from dataclasses import dataclass


@dataclass
class JournalQuestion:
    id: str
    category: str
    text: str


QUESTIONS: list[JournalQuestion] = [
    # ── Goals & Vision ────────────────────────────────────────────
    JournalQuestion("g1", "Goals & Vision",
        "What's one outcome you're working toward right now, and what's one concrete next step you can take?"),
    JournalQuestion("g2", "Goals & Vision",
        "If you could only accomplish three things in the next 90 days, what would they be and why?"),
    JournalQuestion("g3", "Goals & Vision",
        "What does success look like for you in six months? What would need to be true?"),
    JournalQuestion("g4", "Goals & Vision",
        "What's a goal you've been putting off? What's the real reason you haven't started?"),
    JournalQuestion("g5", "Goals & Vision",
        "What are you optimizing your life for right now? Is that actually what you want?"),
    JournalQuestion("g6", "Goals & Vision",
        "What's a long-term aspiration that still excites you even when things are hard?"),

    # ── Challenges & Growth ────────────────────────────────────────
    JournalQuestion("c1", "Challenges & Growth",
        "What obstacle did you face recently? What did it reveal about how you think or work?"),
    JournalQuestion("c2", "Challenges & Growth",
        "Where are you feeling stuck right now, and what would it look like to make even 1% progress?"),
    JournalQuestion("c3", "Challenges & Growth",
        "What's a mistake you made recently? What would you do differently, and what did it teach you?"),
    JournalQuestion("c4", "Challenges & Growth",
        "What's something that's harder than you expected it to be? What has that revealed?"),
    JournalQuestion("c5", "Challenges & Growth",
        "What skill or habit are you actively trying to build? What's your biggest obstacle?"),
    JournalQuestion("c6", "Challenges & Growth",
        "What feedback have you received lately that was uncomfortable but probably true?"),

    # ── Learning & Insights ────────────────────────────────────────
    JournalQuestion("l1", "Learning & Insights",
        "What's something you understood more clearly today or this week? How does it connect to what you already knew?"),
    JournalQuestion("l2", "Learning & Insights",
        "What's an idea or concept you've been thinking about lately that feels important?"),
    JournalQuestion("l3", "Learning & Insights",
        "What did you read, watch, or hear recently that changed how you think about something?"),
    JournalQuestion("l4", "Learning & Insights",
        "What's a belief you held a year ago that you've updated? What changed your mind?"),
    JournalQuestion("l5", "Learning & Insights",
        "What's something you've been curious about that you haven't explored yet? What's holding you back?"),
    JournalQuestion("l6", "Learning & Insights",
        "What problem are you currently trying to solve? What approaches have you already ruled out?"),

    # ── Relationships ──────────────────────────────────────────────
    JournalQuestion("r1", "Relationships",
        "Who did you have a meaningful interaction with recently? What did you take away from it?"),
    JournalQuestion("r2", "Relationships",
        "Who in your life is currently having the biggest positive impact on how you think or grow?"),
    JournalQuestion("r3", "Relationships",
        "Is there a relationship you've been neglecting that matters to you? What would reconnecting look like?"),
    JournalQuestion("r4", "Relationships",
        "Who do you find energizing to be around, and who do you find draining? What's the pattern?"),
    JournalQuestion("r5", "Relationships",
        "What's a conversation you've been avoiding that you probably need to have?"),

    # ── Emotions & Wellbeing ───────────────────────────────────────
    JournalQuestion("e1", "Emotions & Wellbeing",
        "How would you describe your mental state over the past week? What's been driving it?"),
    JournalQuestion("e2", "Emotions & Wellbeing",
        "What's something that's been weighing on you? Is it something you can act on, or something you need to let go of?"),
    JournalQuestion("e3", "Emotions & Wellbeing",
        "When did you last feel genuinely energized and in flow? What were the conditions?"),
    JournalQuestion("e4", "Emotions & Wellbeing",
        "What are you grateful for that you haven't acknowledged lately?"),
    JournalQuestion("e5", "Emotions & Wellbeing",
        "What would feeling more balanced look like in your life right now? What's one thing you could change?"),
    JournalQuestion("e6", "Emotions & Wellbeing",
        "What emotion have you been avoiding or suppressing? What happens when you sit with it?"),

    # ── Reflection & Decisions ─────────────────────────────────────
    JournalQuestion("d1", "Reflection & Decisions",
        "What decision did you make recently that you'd like to think through more carefully?"),
    JournalQuestion("d2", "Reflection & Decisions",
        "What are you currently over-thinking? What would happen if you just made a call?"),
    JournalQuestion("d3", "Reflection & Decisions",
        "What tradeoff are you navigating right now? What matters most to you when you examine it honestly?"),
    JournalQuestion("d4", "Reflection & Decisions",
        "What's one thing you'd tell your past self from six months ago?"),
    JournalQuestion("d5", "Reflection & Decisions",
        "What do you wish you had more clarity on right now? What information would help?"),
    JournalQuestion("d6", "Reflection & Decisions",
        "Looking at how you've spent your time this week — does it align with what you say you care about?"),
]

# Category order for rotation — ensures each session spans different themes
CATEGORY_ORDER = [
    "Goals & Vision",
    "Challenges & Growth",
    "Learning & Insights",
    "Relationships",
    "Emotions & Wellbeing",
    "Reflection & Decisions",
]


def get_session_questions(session_index: int = 0, count: int = 3) -> list[JournalQuestion]:
    """Return `count` questions from different categories for a given session index.

    Uses session_index to rotate which question is pulled from each category,
    so repeated sessions don't repeat the same questions immediately.
    """
    by_category: dict[str, list[JournalQuestion]] = {}
    for q in QUESTIONS:
        by_category.setdefault(q.category, []).append(q)

    selected = []
    for i, category in enumerate(CATEGORY_ORDER[:count]):
        qs = by_category.get(category, [])
        if qs:
            q = qs[session_index % len(qs)]
            selected.append(q)

    return selected
