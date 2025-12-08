# 📘 EduGrade: Retrieval-Augmented Short-Answer Grading
End-to-end system for retrieving scientific context, generating reference answers, and grading student short responses with a RoBERTa classifier.

Project structure includes:
- `final_project.ipynb` — complete pipeline  
- `science_kb_FINAL.json` — knowledge-base passages  
- `science_kb_FINAL.npy` — MPNet embeddings  
- `science_kb_FINAL.faiss` — FAISS index
- `baseline_grader/` — not included due to size; required for full reproduction  

## ⚙️ Environment

The simplest way to run the project is **Google Colab**, since all paths in the notebook are already configured for Colab execution.

If running locally, update the file paths in the notebook.

### Requirements
- Python 3.9+
- transformers  
- datasets  
- sentence-transformers  
- faiss-cpu  
- openai  
- numpy  
- scikit-learn  

Install:
```bash
pip install transformers datasets sentence-transformers faiss-cpu openai scikit-learn
```
## 🚀 Running the Pipeline
1. Open Google Colab.  
2. Upload:
   - `science_kb_FINAL.json`
   - `science_kb_FINAL.npy`
   - `science_kb_FINAL.faiss`
3. Add your OpenAI API key in the corresponding notebook cell.  
4. Run the notebook top-to-bottom.

The pipeline performs:
- retrieval  
- RAG reference generation  
- grading with RoBERTa  
- evaluation (baseline vs RAG)  
- semantic quality scoring for generated references  

---

## 🧩 System Overview

### Retrieval
MPNet encodes questions and FAISS retrieves the most relevant scientific passages.

### RAG Reference Generation
GPT-4o-mini generates a consistent teacher-style reference answer conditioned on the retrieved passage.

### Grading
A RoBERTa classifier predicts:
- correct  
- contradictory  
- incorrect  

Input format:

[QUESTION] ...
[REFERENCE] ...
[ANSWER] ...


## Evaluation Summary

### Baseline Grader
Accuracy approximately 0.60  
Macro-F1 approximately 0.46  

### RAG Pipeline
Shows higher accuracy.  
Macro-F1 decreases slightly because generated references differ stylistically from gold labels.

### RAG Quality
BERTScore (RAG vs Gold) approximately 0.878  
Indicates high semantic similarity even with surface-level variation.

## End-to-End Demonstration
The notebook includes three complete examples showing:
- retrieved passages  
- generated RAG references  
- final predicted labels  

## Reproduction Notes
- All retrieval resources (.json, .npy, .faiss) are included.  
- The RoBERTa grading model is not uploaded due to size constraints.  
- To reproduce classifier predictions exactly, place the baseline_grader folder at the path referenced in the notebook.

## Future Extensions
- Retrain the classifier on RAG-generated references to reduce style mismatch.  
- Add cross-encoder reranking or passage merging to improve retrieval.  
- Expand the knowledge base.  
- Evaluate robustness on real student responses.
