---
title: Banner Flip Engine Prototype
emoji: 🌍
colorFrom: pink
colorTo: green
sdk: docker
pinned: false
short_description: Protype of Banner Flip Engine for AIE5 Midterm
---
# Loom Video
https://www.loom.com/share/207c807c11ca45fbbdba4bf19fc444cd?sid=54d3eff0-152c-46a6-ac7a-85077e2d93a4
# Banner Flip Engine
## AIE5 Midterm Project
## By Lewis Tribble
### Problem and Audience
To counter revenue loss from retail grocery store closures and drive growth, my company needs to quickly and cost-effectively expand its store network by converting competitor stores into banner flip opportunities.

The term **banner flip** refers to a competitor store becoming a Save A Lot store and sourcing their inventory from our distribution centers.  Acquiring competitor stores through banner flips is a faster and more cost-effective growth strategy than building new stores. Our store count has fallen 25% over the last 5 years, with revenue falling at a comparable rate. The business development team has been working on finding banner flip opportunities, but it is an arduous process connecting information from various sources, particularly connecting a store location with the owner/decision makers and their contact information.

### Proposed Solution
An AI-powered Banner Flip Engine chatbot will identify and prioritize competitor stores within the "flip zone" by analyzing various factors such as local market conditions and financial trends. It maps store-to-owner relationships by providing accurate contact information for store owners and decision-makers. The engine ranks stores by their conversion likelihood based on financial signals, lease expiration timelines, and competitive pressures. It also generates tailored outreach materials for regional sales teams, including owner profiles and personalized value propositions for flipping. As a stretch goal, it could offer a self-service research tool where competitor store owners can input their own store data and receive a personalized report on the benefits of switching banners.

For the midterm assignment, the solution is focused on answering questions about competitor stores and generating store profiles that provide demographic facts with valuable insights. These capabilities are the basis for the Banner Flip Engine vision.  

The technology stack includes the following:
- 🤖 LLM: gpt-4o-mini, gpt-4o (for interacting with mutliple tools).  Data security is not an issue because all data is publicly available.
- 🔢 Embedding Models: text-embedding-3-small, snowflake-arctic-embed-l.  Again, data security is not an issue, but it is important to show that open-source models are as effective and can be used when data privicy is required.
- 🎺 Orchestration: LangGraph. The architecture  (specialist agents + supervisor + stateful workflow) aligns perfectly with LangGraph's design purpose for LLM-powered applications.
- ↗️ Vector Store: Qdrant. It's fast, scalable, flexible, and FREE. And we've used it exclusively to date in this course.
- 📈 Monitoring: LangSmith. Tight integration with LangChain while supporting custom LLM implementations.
- 📐 Evaluation: RAGAS. RAGAS is particularly valuable because it addresses the unique challenge of evaluating both retrieval and generation components simultaneously, which traditional NLP metrics fail to capture effectively.
- 💬 User Interface: Chainlit. Easy to implement quickly. I tried Streamlit, but I found it less intuitive and would have taken longer than using what I'm already familiar with through this course.
- 🛎️ Inference & Serving: Hugging Face. Required (?) for this assignment. Familiarity - other options would have taken considerable time to learn.
### Dealing with Data
Our business devlopment team purchased competitor data from a vendor that specializes in this. They provided me with a spreadsheet (20k rows, 30 columns, 16MB in csv format)... we have not addressed structured data in this course, so I did have to spend time researching and trying diffeent methods.  I did create an SQL agent that easily handles the full dataset and provides excellent responses. But RAG using embeddings is required for this assignment. I created utilities to convert CVS to JSON, then generating semantic content from JSON. I realized, after some processing attempts, that the getting the full dataset into a vector store takes way too long and costs way too much (using OpenAI embeddings) for the purposes of this protype. So I used only the first 200 rows for RAG.

When generating semantic content from the structured data, I put a seperator ("="*80) between the store entries, which I used for chunking, making each store a chunk (all are under 1000 tokens). I found this to be effective, particularly with summarization, but is unable to handle query's like "How many stores are in Florida?", which the SQL agent handles with ease.

For creating store profiles, I used US Census Bureau's API service to get vast array of demographic data. There are many business profiling API providers, but I did not want to incurr those costs for this prototype project. Unfortunately there are many gaps in the free census data, so the resulting profiles are generally not useful... but it shows that this approach does work.
### Ene-to-End Prototype
https://huggingface.co/spaces/philocifer/banner_flip_engine_prototype
### Golden Test Dataset

| Metric                      | Score   |
|-----------------------------|---------|
| Context Recall              | 0.7867  |
| Faithfulness                | 0.8726  |
| Factual Correctness         | 0.5185  |
| Answer Relevancy            | 0.9069  |
| Context Entity Recall       | 0.4353  |
| Noise Sensitivity Relevant  | 0.5952  |

Strengths:
1. High Answer Relevancy (0.9069) - Responses stay focused on user queries
2. Strong Faithfulness (0.8726) - Answers remain grounded in provided context
3. Good Context Recall (0.7867) - Retrieves most relevant context chunks

Weaknesses:
1. Low Factual Correctness (0.5185) - Generated answers contain factual errors despite good context
2. Poor Entity Recall (0.4353) - Struggles to identify/store key entities (store names, locations)
3. Noise Sensitivity (0.5952) - Vulnerable to irrelevant/conflicting information
### Fine-Tuning Open-Source Embeddings
https://huggingface.co/philocifer/banner-flip-arctic-embed-l
### Assessing Performance of Fine-Tuned Embeddings
| Metric                      | Score   |
|-----------------------------|---------|
| Context Recall              | 0.9175  |
| Faithfulness                | 0.8203  |
| Factual Correctness         | 0.7225  |
| Answer Relevancy            | 0.9669  |
| Context Entity Recall       | 0.5711  |
| Noise Sensitivity Relevant  | 0.0000  |

#### Evaluation Comparison
Significant Improvements
- Factual Correctness surged 39% (0.52 → 0.72) - Substantially more reliable answers
- Context Recall jumped 16% (0.79 → 0.92) - Better retrieval of relevant information
- Answer Relevancy reached near-perfect 0.97 (+7%) - Sharper focus on query intent

Trade-offs
- Faithfulness dipped 6% (0.87 → 0.82) - Slightly less strict adherence to source context despite better facts

Notable Changes
- Noise Sensitivity collapsed to 0.00 (-100%) - Complete immunity to irrelevant information (requires verification)
- Entity Recognition improved 31% (0.44 → 0.57) - Remains a relative weakness in the system

In the second half of the course, I will focus more on improving the SQL agent as it is much better at handling structured data in large volumes.
