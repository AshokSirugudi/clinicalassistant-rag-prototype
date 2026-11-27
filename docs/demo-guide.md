# Demo guide

I start with Set A in an interview. It shows a visit citation, a shared source, and a refusal using the same question under two different patients. Set B gives me three more examples from the appointment history.

Leave the app in **Offline Keyword Search**. Select the listed patient, paste the exact question, and click **Ask**. The excerpts below were verified through the root Streamlit launcher on GitHub `main` at `54edc51fdb4f3af35b2f970c1d99000748f75d8a` with `top_k=5`. This demo needs no OpenAI key or vector index. [Setup instructions](../README.md#run-it-in-codex-cloud)

## Set A

### 1. P001 · Knee discomfort

**Ask** “What does the note say about knee discomfort and limits on usual activities?”

> Activity-related knee discomfort with some limits on usual activities.

Expect **Visit note**, section **Symptoms**, appointment `A001`, dated 2026-10-10 with Iris Mosswick. [A001.md](../data/patients/visit_notes/P001/A001.md#symptoms)

### 2. P003 · Shared headache guidance

**Ask** “For a headache diary, what should I note about how long each headache lasted, how intense it felt, and which activities it interrupted?”

> For each headache, note when it started, how long it lasted, how intense it felt, and which activities it interrupted.

Expect **General guidance**, section **What to Record**, with no appointment details. This is the shared diary guide available across patient selections. [headache-diary.md](../data/shared/headache-diary.md#what-to-record)

### 3. P010 · Footwear and isolation check

**Ask** “What about footwear?”

> The visit was a routine foot examination. The patient raised footwear concerns.

Expect **Visit note**, section **Symptoms**, appointment `A037`, dated 2026-10-19 with Rhea Crestwell. [A037.md](../data/patients/visit_notes/P010/A037.md#symptoms)

Switch to **P001**. The previous question and citation should clear. Ask “What about footwear?” again. Expect “No matching documents were found for the selected patient or shared guidance.” The response should contain no excerpt or citation.

## Set B

### 4. P002 · Blood pressure monitoring

**Ask** “What was recommended about recording home readings and bringing the log to the next blood pressure visit?”

> Continue recording home readings and bring the log to the next blood pressure visit.

Expect **Visit note**, section **Recommendations**, appointment `A006`, dated 2026-05-11 with Owen Birchmere. [A006.md](../data/patients/visit_notes/P002/A006.md#recommendations)

### 5. P006 · Reflux symptom tracking

**Ask** “What should I track about meal timing and symptom frequency for the reflux follow-up?”

> Track meal timing and symptom frequency for the reflux follow-up.

Expect **Visit note**, section **Recommendations**, appointment `A021`, dated 2026-10-15 with Idris Pinewell. [A021.md](../data/patients/visit_notes/P006/A021.md#recommendations)

### 6. P008 · Exercise-related cough

**Ask** “What did the note say to record about when coughing starts during practice?”

> Record when coughing starts during practice and bring those observations to the cough follow-up.

Expect **Visit note**, section **Recommendations**, appointment `A029`, dated 2026-10-17 with Nadia Elmford. [A029.md](../data/patients/visit_notes/P008/A029.md#recommendations)

The demo uses literal word matching over fictional data. Phrasing changes can affect results. I use the citations to show what the source says, and I leave clinical decisions outside the scope of the demo.
