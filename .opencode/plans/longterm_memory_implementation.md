# Long-Term Memory Implementation Plan - ai_agent_ha

**Date**: 2026-02-05  
**Source**: OpenClaw memory-lancedb analysis via Gemini 3 Pro  
**Status**: Ready for Implementation  
**Priority**: HIGH - Major feature gap

---

## Executive Summary

OpenClaw implements a sophisticated **Long-Term Memory (LTM)** system using:
- **LanceDB** - embedded vector database
- **Auto-Capture** - automatically saves important facts from conversation
- **Auto-Recall** - injects relevant memories before agent responds
- **Tools** - agent & user can manage memories

**ai_agent_ha currently lacks this** - all knowledge is lost between sessions.

**Impact**: Users must repeat preferences every conversation.

---

## Problem Statement

### Current State in ai_agent_ha

**Session-only memory**:
- Conversation history stored temporarily
- Knowledge lost after restart
- User must repeat:
  - "I prefer detailed explanations"
  - "My bedroom light is light.bedroom_lamp"
  - "Always turn on heater at 6 AM"

**RAG is not Long-Term Memory**:
- ai_agent_ha's RAG indexes **Home Assistant entities**
- Doesn't store **user preferences** or **conversation facts**
- No mechanism to remember "what user told me yesterday"

---

## OpenClaw LTM Architecture

### 1. Storage Layer

**Database**: LanceDB (embedded, file-based vector DB)  
**Location**: `~/.openclaw/memory/lancedb`  
**Schema**:

```typescript
interface MemoryEntry {
  id: string;           // UUID
  text: string;         // The fact (e.g., "User prefers concise answers")
  vector: number[];     // Embedding (1536 floats for OpenAI)
  importance: number;   // 0.0 - 1.0 weight
  category: "preference" | "fact" | "decision" | "entity" | "other";
  createdAt: number;    // Unix timestamp
  metadata?: {
    source?: string;    // "user" | "agent" | "auto-capture"
    session?: string;   // Session ID where captured
  };
}
```

**Lifecycle**:
```typescript
// Init (lazy)
async function ensureInitialized() {
  if (!db) {
    db = await lancedb.connect(dbPath);
    table = await db.openTable("memories");
  }
}

// Store
async function store(text: string, category: string, importance: number) {
  const vector = await embeddings.embed(text);
  
  // Deduplication check
  const similar = await search(vector, { limit: 1, minScore: 0.95 });
  if (similar.length > 0) {
    console.log("Duplicate memory, skipping");
    return;
  }
  
  await table.add([{
    id: uuid(),
    text,
    vector,
    importance,
    category,
    createdAt: Date.now()
  }]);
}

// Search
async function search(queryVector: number[], options: { limit: number, minScore: number }) {
  const results = await table
    .vectorSearch(queryVector)
    .limit(options.limit)
    .execute();
  
  // Filter by score threshold
  return results.filter(r => (1 - r._distance) >= options.minScore);
}
```

---

### 2. Auto-Capture Mechanism

**Trigger**: Hook `agent_end` (after agent finishes response)

**Flow**:
```typescript
api.on("agent_end", async (event) => {
  const messages = event.messages;
  
  // 1. Extract candidates
  const candidates = [];
  for (const msg of messages) {
    if (msg.role !== "user" && msg.role !== "assistant") continue;
    if (msg.content.length < 10 || msg.content.length > 500) continue;
    
    // Check triggers
    if (shouldCapture(msg.content)) {
      candidates.push(msg.content);
    }
  }
  
  // 2. Limit to 3 per turn
  const toCapture = candidates.slice(0, 3);
  
  // 3. Store each
  for (const text of toCapture) {
    const category = detectCategory(text);
    await store(text, category, 0.7);
  }
});
```

**Trigger Patterns**:
```typescript
const MEMORY_TRIGGERS = [
  // Explicit commands
  /remember|zapamiętaj|zapisz/i,
  
  // Preferences
  /I prefer|preferuję|wolę/i,
  /I like|lubię|podoba mi się/i,
  /I don't like|nie lubię/i,
  /always|zawsze/i,
  /never|nigdy/i,
  
  // Personal info
  /my name is|nazywam się/i,
  /I live in|mieszkam w/i,
  /my (\\w+) is/i,  // "my email is", "my phone is"
  
  // Decisions
  /I decided|zdecydowałem/i,
  /we agreed|ustaliliśmy/i,
  /let's use|użyjmy/i,
  
  // Entity references
  /\\S+@\\S+\\.\\S+/,  // Email
  /\\+?\\d{9,}/,       // Phone number
];
```

**Category Detection**:
```typescript
function detectCategory(text: string): string {
  const lower = text.toLowerCase();
  
  if (lower.match(/prefer|like|love|enjoy|hate/)) {
    return "preference";
  }
  
  if (lower.match(/decided|agreed|will use|chose/)) {
    return "decision";
  }
  
  if (lower.match(/@|\\+\\d{9}/)) {
    return "entity";  // Contact info
  }
  
  return "fact";
}
```

**Sanitization** (prevent loops):
```typescript
function shouldCapture(text: string): boolean {
  // Don't capture if text is from memory injection
  if (text.includes("<relevant-memories>")) {
    return false;
  }
  
  // Check trigger patterns
  return MEMORY_TRIGGERS.some(pattern => pattern.test(text));
}
```

---

### 3. Auto-Recall Mechanism

**Trigger**: Hook `before_agent_start` (before agent processes query)

**Flow**:
```typescript
api.on("before_agent_start", async (event) => {
  const userPrompt = event.prompt;
  
  // 1. Generate embedding for query
  const queryVector = await embeddings.embed(userPrompt);
  
  // 2. Search LTM
  const memories = await search(queryVector, {
    limit: 3,
    minScore: 0.3  // 30% similarity threshold
  });
  
  // 3. Inject into context
  if (memories.length > 0) {
    const memoryText = memories
      .map(m => `- [${m.category}] ${m.text}`)
      .join('\\n');
    
    return {
      prependContext: `<relevant-memories>
The following memories may be relevant to this conversation:
${memoryText}
</relevant-memories>

`
    };
  }
});
```

**Example Output**:
```
<relevant-memories>
The following memories may be relevant to this conversation:
- [preference] User prefers detailed technical explanations
- [fact] User's main bedroom light is light.bedroom_main
- [decision] User wants morning automation at 6 AM
</relevant-memories>

User: Turn on my bedroom light
Assistant: [sees memory, knows which light]
```

**Scoring**:
```typescript
// LanceDB returns distance (0 = identical, 2 = opposite)
// Convert to similarity score (0-1)
const score = 1 - (distance / 2);

// Filter by threshold
if (score >= 0.3) {  // Keep if >30% similar
  // Include in results
}
```

---

### 4. User Interface

#### A. Tools (for Agent)

**Tool: memory_store**
```typescript
{
  name: "memory_store",
  description: "Store a fact, preference, or decision in long-term memory",
  parameters: {
    type: "object",
    properties: {
      text: {
        type: "string",
        description: "The information to remember"
      },
      category: {
        type: "string",
        enum: ["preference", "fact", "decision", "entity", "other"],
        description: "Type of memory"
      },
      importance: {
        type: "number",
        description: "How important (0.0-1.0), default 0.7"
      }
    },
    required: ["text"]
  }
}
```

**Tool: memory_recall**
```typescript
{
  name: "memory_recall",
  description: "Search long-term memory for relevant information",
  parameters: {
    type: "object",
    properties: {
      query: {
        type: "string",
        description: "What to search for"
      },
      limit: {
        type: "number",
        description: "Max results, default 5"
      }
    },
    required: ["query"]
  }
}
```

**Tool: memory_forget**
```typescript
{
  name: "memory_forget",
  description: "Delete a memory by ID or search term",
  parameters: {
    type: "object",
    properties: {
      id: {
        type: "string",
        description: "Memory ID to delete (if known)"
      },
      query: {
        type: "string",
        description: "Search for memory to delete"
      }
    }
  }
}
```

#### B. CLI Commands

**List memories**:
```bash
ha ai_agent_ha.memory_list
# Output:
# Total memories: 42
# Categories:
#   - preference: 15
#   - fact: 20
#   - decision: 5
#   - entity: 2
```

**Search memories**:
```bash
ha ai_agent_ha.memory_search --query "bedroom light"
# Output:
# [fact] User's bedroom light is light.bedroom_main (score: 0.85)
# [preference] User prefers warm white light in bedroom (score: 0.72)
```

**Delete memory**:
```bash
ha ai_agent_ha.memory_delete --id "mem_abc123"
# or
ha ai_agent_ha.memory_delete --query "my email"
```

---

### 5. Smart Features

#### A. Deduplication

**Before storing**, check if similar memory exists:

```python
async def store_memory(text: str, category: str, importance: float):
    # Generate embedding
    vector = await embedding_provider.embed(text)
    
    # Search for duplicates
    similar = await lancedb_table.search(vector).limit(1).to_list()
    
    if similar and similar[0]['_distance'] < 0.05:  # 95% similarity
        _LOGGER.info(f"Duplicate memory: {text}")
        return None  # Skip
    
    # Store
    await lancedb_table.add([{
        'id': str(uuid4()),
        'text': text,
        'vector': vector,
        'importance': importance,
        'category': category,
        'created_at': time.time()
    }])
```

#### B. Importance Scoring

**Dynamic importance** based on context:

```python
def calculate_importance(text: str, category: str, source: str) -> float:
    base = 0.5
    
    # User explicitly said "remember" -> high importance
    if 'remember' in text.lower() or 'zapamiętaj' in text.lower():
        base += 0.3
    
    # Preferences are important
    if category == 'preference':
        base += 0.2
    
    # Decisions are very important
    if category == 'decision':
        base += 0.3
    
    # Auto-captured facts are less important
    if source == 'auto-capture':
        base -= 0.1
    
    return min(max(base, 0.0), 1.0)
```

#### C. Auto-Categorization

```python
def detect_category(text: str) -> str:
    text_lower = text.lower()
    
    # Preference indicators
    if any(word in text_lower for word in ['prefer', 'like', 'love', 'hate', 'enjoy', 'wolę', 'lubię']):
        return 'preference'
    
    # Decision indicators
    if any(word in text_lower for word in ['decided', 'agreed', 'chose', 'will use', 'zdecydowałem']):
        return 'decision'
    
    # Entity indicators (email, phone)
    if re.search(r'\\S+@\\S+\\.\\S+', text) or re.search(r'\\+?\\d{9,}', text):
        return 'entity'
    
    return 'fact'
```

#### D. GDPR Compliance

**Forget specific data**:

```python
async def forget_memory(query: str = None, id: str = None):
    if id:
        # Delete by ID
        await lancedb_table.delete(f"id = '{id}'")
    elif query:
        # Search and delete
        vector = await embedding_provider.embed(query)
        results = await lancedb_table.search(vector).limit(5).to_list()
        
        for result in results:
            if result['_distance'] < 0.3:  # Only delete if very similar
                await lancedb_table.delete(f"id = '{result['id']}'")
    
    _LOGGER.info(f"Deleted memories matching: {query or id}")
```

---

## Implementation Plan for ai_agent_ha

### Phase 1: Storage Layer (2-3 hours)

**Goal**: Set up LanceDB storage

**Files to Create**:
1. `custom_components/ai_agent_ha/memory/lancedb_store.py`

```python
import lancedb
import uuid
from typing import List, Dict, Any, Optional

class LongTermMemoryStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.db = None
        self.table = None
    
    async def initialize(self):
        """Lazy init."""
        if self.db is None:
            self.db = await lancedb.connect_async(self.db_path)
            
            # Create table if not exists
            try:
                self.table = await self.db.open_table("memories")
            except:
                # Table doesn't exist, create it
                self.table = await self.db.create_table(
                    "memories",
                    data=[{
                        'id': 'init',
                        'text': 'Initialization',
                        'vector': [0.0] * 1536,
                        'importance': 0.0,
                        'category': 'other',
                        'created_at': 0
                    }]
                )
                # Delete init row
                await self.table.delete("id = 'init'")
    
    async def store(self, text: str, vector: List[float], category: str, importance: float) -> str:
        """Store memory."""
        await self.initialize()
        
        # Check for duplicates
        similar = await self.search(vector, limit=1, min_score=0.95)
        if similar:
            _LOGGER.debug(f"Duplicate memory: {text}")
            return None
        
        # Add
        memory_id = str(uuid.uuid4())
        await self.table.add([{
            'id': memory_id,
            'text': text,
            'vector': vector,
            'importance': importance,
            'category': category,
            'created_at': time.time()
        }])
        
        return memory_id
    
    async def search(self, query_vector: List[float], limit: int = 5, min_score: float = 0.3) -> List[Dict]:
        """Vector search."""
        await self.initialize()
        
        results = await self.table.search(query_vector).limit(limit).to_list()
        
        # Convert distance to score and filter
        filtered = []
        for r in results:
            score = 1 - (r['_distance'] / 2)  # Normalize to 0-1
            if score >= min_score:
                filtered.append({
                    'id': r['id'],
                    'text': r['text'],
                    'category': r['category'],
                    'importance': r['importance'],
                    'score': score,
                    'created_at': r['created_at']
                })
        
        return filtered
    
    async def delete(self, memory_id: str):
        """Delete by ID."""
        await self.initialize()
        await self.table.delete(f"id = '{memory_id}'")
    
    async def count(self) -> int:
        """Total memories."""
        await self.initialize()
        result = await self.table.count_rows()
        return result
```

**Dependencies**:
```bash
pip install lancedb
```

**Test**:
```python
store = LongTermMemoryStore("/config/.storage/ai_agent_ha/ltm")
await store.initialize()

# Store
vec = await embeddings.embed("User prefers detailed answers")
await store.store("User prefers detailed answers", vec, "preference", 0.8)

# Search
results = await store.search(vec, limit=3)
assert len(results) == 1
assert results[0]['text'] == "User prefers detailed answers"
```

---

### Phase 2: Auto-Capture (3-4 hours)

**Goal**: Automatically save facts from conversation

**Files to Create**:
2. `custom_components/ai_agent_ha/memory/auto_capture.py`

```python
import re
from typing import List, Dict

# Trigger patterns
MEMORY_TRIGGERS = [
    # Explicit
    r'remember|zapamiętaj|zapisz|save',
    
    # Preferences
    r'I prefer|preferuję|wolę',
    r'I like|lubię|podoba mi się',
    r'I don\'t like|nie lubię',
    r'always|zawsze',
    r'never|nigdy',
    
    # Personal
    r'my name is|nazywam się',
    r'I live in|mieszkam w',
    r'my (\\w+) is',
    
    # Decisions
    r'I decided|zdecydowałem',
    r'we agreed|ustaliliśmy',
    r'let\'s use|użyjmy',
]

def should_capture(text: str) -> bool:
    """Check if text should be captured."""
    # Don't capture if from memory injection
    if '<relevant-memories>' in text:
        return False
    
    # Check patterns
    text_lower = text.lower()
    return any(re.search(pattern, text_lower, re.IGNORECASE) for pattern in MEMORY_TRIGGERS)

def detect_category(text: str) -> str:
    """Detect memory category."""
    text_lower = text.lower()
    
    if any(word in text_lower for word in ['prefer', 'like', 'love', 'hate', 'wolę']):
        return 'preference'
    
    if any(word in text_lower for word in ['decided', 'agreed', 'chose', 'zdecydowałem']):
        return 'decision'
    
    if re.search(r'\\S+@\\S+\\.\\S+', text):
        return 'entity'
    
    return 'fact'

async def auto_capture_from_conversation(
    messages: List[Dict],
    ltm_store: LongTermMemoryStore,
    embedding_provider
):
    """Scan conversation and auto-capture facts."""
    
    candidates = []
    
    for msg in messages:
        if msg['role'] not in ['user', 'assistant']:
            continue
        
        content = msg['content']
        
        # Filter
        if len(content) < 10 or len(content) > 500:
            continue
        
        if should_capture(content):
            candidates.append(content)
    
    # Limit to 3 per turn
    to_capture = candidates[:3]
    
    # Store
    for text in to_capture:
        category = detect_category(text)
        importance = 0.7  # Default for auto-capture
        
        vector = await embedding_provider.get_embeddings([text])
        await ltm_store.store(text, vector[0], category, importance)
        
        _LOGGER.info(f"Auto-captured [{category}]: {text[:50]}...")
```

**Hook Integration** (in `query_processor.py`):

```python
# After agent response
async def process_query(self, query: str, **kwargs):
    # ... existing processing ...
    
    # Auto-capture (if LTM enabled)
    if self.ltm_store:
        await auto_capture_from_conversation(
            messages=kwargs.get('messages', []),
            ltm_store=self.ltm_store,
            embedding_provider=self.embedding_provider
        )
    
    return result
```

---

### Phase 3: Auto-Recall (2-3 hours)

**Goal**: Inject relevant memories before agent responds

**Files to Modify**:
3. `custom_components/ai_agent_ha/core/query_processor.py`

```python
async def process_query(self, query: str, **kwargs):
    # Auto-recall BEFORE processing
    if self.ltm_store:
        memories = await self.auto_recall(query)
        
        if memories:
            # Inject into system prompt
            memory_context = self._format_memories(memories)
            
            system_prompt = kwargs.get('system_prompt', '')
            kwargs['system_prompt'] = memory_context + '\\n\\n' + system_prompt
    
    # ... rest of processing ...

async def auto_recall(self, query: str) -> List[Dict]:
    """Search LTM for relevant memories."""
    # Generate embedding
    query_vector = await self.embedding_provider.get_embeddings([query])
    
    # Search
    memories = await self.ltm_store.search(
        query_vector[0],
        limit=3,
        min_score=0.3
    )
    
    if memories:
        _LOGGER.info(f"Recalled {len(memories)} memories")
    
    return memories

def _format_memories(self, memories: List[Dict]) -> str:
    """Format memories for injection."""
    lines = ["<relevant-memories>"]
    lines.append("The following memories may be relevant:")
    
    for mem in memories:
        lines.append(f"- [{mem['category']}] {mem['text']}")
    
    lines.append("</relevant-memories>")
    
    return '\\n'.join(lines)
```

---

### Phase 4: Tools & UI (2-3 hours)

**Goal**: Let agent & user manage memories

**Files to Create**:
4. `custom_components/ai_agent_ha/tools/memory.py`

```python
# Tool definitions
MEMORY_TOOLS = [
    {
        "name": "memory_store",
        "description": "Store information in long-term memory",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "What to remember"},
                "category": {
                    "type": "string",
                    "enum": ["preference", "fact", "decision", "entity", "other"]
                },
                "importance": {"type": "number", "description": "0.0-1.0"}
            },
            "required": ["text"]
        }
    },
    {
        "name": "memory_recall",
        "description": "Search long-term memory",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "number"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "memory_forget",
        "description": "Delete a memory",
        "parameters": {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "query": {"type": "string"}
            }
        }
    }
]

# Tool handlers
async def handle_memory_store(args: Dict) -> str:
    text = args['text']
    category = args.get('category', 'fact')
    importance = args.get('importance', 0.7)
    
    vector = await embedding_provider.embed(text)
    memory_id = await ltm_store.store(text, vector, category, importance)
    
    return f"Stored memory: {memory_id}"

async def handle_memory_recall(args: Dict) -> str:
    query = args['query']
    limit = args.get('limit', 5)
    
    vector = await embedding_provider.embed(query)
    results = await ltm_store.search(vector, limit=limit)
    
    if not results:
        return "No relevant memories found"
    
    return json.dumps(results, indent=2)

async def handle_memory_forget(args: Dict) -> str:
    if 'id' in args:
        await ltm_store.delete(args['id'])
        return f"Deleted memory: {args['id']}"
    elif 'query' in args:
        vector = await embedding_provider.embed(args['query'])
        results = await ltm_store.search(vector, limit=5, min_score=0.8)
        
        deleted = []
        for r in results:
            await ltm_store.delete(r['id'])
            deleted.append(r['text'])
        
        return f"Deleted {len(deleted)} memories: {deleted}"
```

**Home Assistant Services** (in `__init__.py`):

```python
# Register HA services
hass.services.async_register(
    DOMAIN,
    'memory_list',
    async_handle_memory_list
)

hass.services.async_register(
    DOMAIN,
    'memory_search',
    async_handle_memory_search
)

hass.services.async_register(
    DOMAIN,
    'memory_delete',
    async_handle_memory_delete
)

async def async_handle_memory_list(call):
    """List all memories."""
    count = await ltm_store.count()
    return {'total': count}

async def async_handle_memory_search(call):
    """Search memories."""
    query = call.data.get('query')
    vector = await embedding_provider.embed(query)
    results = await ltm_store.search(vector, limit=10)
    return {'results': results}

async def async_handle_memory_delete(call):
    """Delete memory."""
    memory_id = call.data.get('id')
    await ltm_store.delete(memory_id)
    return {'deleted': memory_id}
```

---

## Testing Plan

### Test 1: Auto-Capture Works
```yaml
# User conversation
User: "Zapamiętaj że wolę szczegółowe odpowiedzi"
Agent: "Zapisane!"

# Verify in DB
Service: ai_agent_ha.memory_list
# Should show 1 memory

Service: ai_agent_ha.memory_search
data:
  query: "odpowiedzi"
# Should return: [preference] Wolę szczegółowe odpowiedzi
```

### Test 2: Auto-Recall Works
```yaml
# New session
User: "Co wolę?"
Agent: [should reference memory] "Widzę że preferujesz szczegółowe odpowiedzi."
```

### Test 3: Deduplication Works
```yaml
# Repeat
User: "Zapamiętaj że wolę szczegółowe odpowiedzi"

# Should NOT create duplicate
Service: ai_agent_ha.memory_list
# Still shows 1 memory
```

---

## Comparison: OpenClaw vs ai_agent_ha

| Feature | OpenClaw | ai_agent_ha (current) | After Implementation |
|---------|----------|----------------------|---------------------|
| Long-Term Storage | ✅ LanceDB | ❌ No LTM | ✅ LanceDB |
| Auto-Capture | ✅ Regex triggers | ❌ None | ✅ Regex triggers |
| Auto-Recall | ✅ Before agent | ❌ None | ✅ Before query |
| Tools for Agent | ✅ 3 tools | ❌ None | ✅ 3 tools |
| User Management | ✅ CLI | ❌ None | ✅ HA Services |
| Deduplication | ✅ 95% threshold | ❌ N/A | ✅ 95% threshold |
| Categories | ✅ 5 types | ❌ N/A | ✅ 5 types |

---

## Timeline

| Phase | Effort | Duration |
|-------|--------|----------|
| Phase 1: Storage | 2-3 hours | Half day |
| Phase 2: Auto-Capture | 3-4 hours | Half day |
| Phase 3: Auto-Recall | 2-3 hours | Half day |
| Phase 4: Tools & UI | 2-3 hours | Half day |
| **Testing** | 2 hours | - |
| **TOTAL** | **11-15 hours** | **2-3 days** |

---

## Expected Impact

**Before (Current)**:
- User: "Wolę krótkie odpowiedzi"
- [Next session]
- User: "Wolę krótkie odpowiedzi" (must repeat!)

**After (With LTM)**:
- User: "Wolę krótkie odpowiedzi"
- Agent: [auto-captures]
- [Next session]
- User: "Co słychać?"
- Agent: [auto-recalls preference, gives short answer]

**User Experience**:
- Feels like agent "knows" you
- No need to repeat preferences
- Continuity across sessions
- Natural conversation flow

---

## Conclusion

Long-Term Memory is a **game-changer** for ai_agent_ha:

1. **Persistence** - knowledge survives restarts
2. **Personalization** - agent adapts to user
3. **Efficiency** - no repeating yourself
4. **Smart** - auto-capture & auto-recall are invisible

OpenClaw's implementation is **production-ready** and can be ported to ai_agent_ha with minimal changes.

**Ready to implement** - all technical details specified.

---

**End of Plan**
