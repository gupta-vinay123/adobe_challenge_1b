# Approach Explanation: Persona-Driven Section Extraction & Ranking

## Overview
This solution is designed to extract and rank the most relevant sections from a collection of PDF documents, tailored to a specific persona and task. The pipeline leverages state-of-the-art models for document layout analysis, text extraction, semantic embedding, and summarization, all orchestrated to work offline and efficiently within a Docker container.

## Pipeline Steps

### 1. Document Layout Analysis (YOLO)
Each PDF is first converted into images (one per page). A YOLOv8 object detection model, fine-tuned to recognize document structures such as titles and section headers, is applied to each page image. This model outputs bounding boxes for key structural elements, enabling precise localization of important sections.

### 2. OCR-Based Text Extraction
For each detected section (e.g., Title, Section-header), the corresponding image region is cropped and passed to RapidOCR, a fast and accurate OCR engine. This step extracts the textual content from each detected section, even in scanned or visually complex documents.

### 3. Section Metadata Collection
For every detected section, the pipeline records metadata including the section label, extracted text, page number, document name, and the full raw text of the page. This information is essential for downstream ranking and summarization.

### 4. Persona-Driven Semantic Ranking
To rank sections by relevance to the persona and task, the system constructs a query string from the persona's role and the job-to-be-done. Both the query and each section's raw text are embedded using a Sentence Transformers model. Cosine similarity is computed between the query embedding and each section embedding, and the top 5 most relevant sections are selected, ensuring uniqueness by section title and document.

### 5. Summarization
For each top-ranked section, the full page text is summarized using a T5-based summarization model. This provides a concise, context-aware summary tailored to the persona's needs.

### 6. Output Formatting
The final output includes metadata, a list of extracted and ranked sections, and their corresponding summaries. The results are written as a structured JSON file to the specified output directory.

## Offline, Reproducible, and Containerized
All models and dependencies are bundled within the Docker image, ensuring fully offline operation. The pipeline is robust to missing files and provides clear error messages for missing models or inputs. Input PDFs and JSON are read from `/app/input`, and results are written to `/app/output`, making integration and automation straightforward.

## Conclusion
This approach combines modern computer vision, NLP, and information retrieval techniques to deliver accurate, persona-driven document analysis, all within a reproducible and portable container environment. 