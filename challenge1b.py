import os
import json
import torch
import fitz
from PIL import Image
from datetime import datetime
from transformers import AutoTokenizer, AutoModel, T5ForConditionalGeneration, T5Tokenizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import torch.nn.functional as F
import cv2
import argparse
from rapidocr_onnxruntime import RapidOCR
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# Load embedding model (update path if needed)
tokenizer = AutoTokenizer.from_pretrained("./embedding_model")
model = AutoModel.from_pretrained("./embedding_model")

# Initialize RapidOCR once
g_ocr = RapidOCR()

YOLO_MODEL_PATH = "yolo_model.pt"
yolo_model = None
if os.path.exists(YOLO_MODEL_PATH):
    yolo_model = YOLO(YOLO_MODEL_PATH)
else:
    print(f"Warning: YOLO model '{YOLO_MODEL_PATH}' not found at import time.")

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def embed_text(text):
    encoded_input = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        model_output = model(**encoded_input)
    embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    return F.normalize(embeddings, p=2, dim=1).numpy()

# Summarization model
tokenizer_summary = T5Tokenizer.from_pretrained("./summarizer_model")
model_summary = T5ForConditionalGeneration.from_pretrained("./summarizer_model")

def summarize_text(text, num_sentences=8):
    if not text.strip():
        return ""
    inputs = tokenizer_summary("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model_summary.generate(
        inputs["input_ids"],
        max_length=num_sentences * 30,  # More words per sentence
        min_length=num_sentences * 15,  # More minimum words
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    return tokenizer_summary.decode(summary_ids[0], skip_special_tokens=True)

def convert_pdf_to_images(pdf_path, dpi=150):
    doc = fitz.open(pdf_path)
    images = []
    for page in doc:
        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    return images

def extract_text_from_bbox(image, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    cropped = image[y1:y2, x1:x2]
    # Use RapidOCR for OCR
    result, _ = g_ocr(cropped)
    # result is a list of [ [text, ...], ... ]
    if result and len(result) > 0:
        # Join all detected text lines
        return ' '.join([line[1] for line in result if len(line) > 1]).strip()
    return ""

def run_yolo_on_image(model, image):
    results = model(image)
    return results[0]

def process_pdf_sections(pdf_path, yolo_model):
    images = convert_pdf_to_images(pdf_path)
    section_data = []
    for page_number, pil_img in enumerate(images, start=1):
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        result = run_yolo_on_image(yolo_model, cv_img)
        page_text = fitz.open(pdf_path)[page_number - 1].get_text()
        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            label = yolo_model.names[cls_id]
            if label in ["Title", "Section-header"]:
                bbox = box.xyxy[0].tolist()
                text = extract_text_from_bbox(cv_img, bbox)
                if text:
                    section_data.append({
                        "level": label,
                        "text": text,
                        "page": page_number,
                        "document": Path(pdf_path).name,
                        "raw_text": page_text
                    })
    return section_data

def rank_sections(section_data, persona_query):
    query_emb = embed_text(persona_query)
    output = []
    for sec in section_data:
        sec_emb = embed_text(sec["raw_text"])
        score = cosine_similarity(sec_emb, query_emb)[0][0]
        sec["score"] = float(score)
        output.append(sec)
    output = sorted(output, key=lambda x: -x["score"])
    unique = []
    seen = set()
    for sec in output:
        key = (sec["document"], sec["text"].strip().lower())
        if key not in seen:
            seen.add(key)
            unique.append(sec)
        if len(unique) == 5:
            break
    for i, sec in enumerate(unique):
        sec["importance_rank"] = i + 1
    return unique

def build_output(input_data, sections):
    metadata = {
        "input_documents": [doc["filename"] for doc in input_data["documents"]],
        "persona": input_data["persona"]["role"],
        "job_to_be_done": input_data["job_to_be_done"]["task"],
        "processing_timestamp": datetime.now().isoformat()
    }
    extracted_sections = []
    subsection_analysis = []
    for sec in sections:
        extracted_sections.append({
            "document": sec["document"],
            "section_title": sec["text"],
            "importance_rank": sec["importance_rank"],
            "page_number": sec["page"]
        })
        subsection_analysis.append({
            "document": sec["document"],
            "refined_text": summarize_text(sec["raw_text"]),
            "page_number": sec["page"]
        })
    output = {
        "metadata": metadata,
        "extracted_sections": extracted_sections,
        "subsection_analysis": subsection_analysis
    }
    return json.dumps(output, indent=4)

def parse_documents(input_json, output_file=None):
    global yolo_model
    if yolo_model is None:
        raise FileNotFoundError(f"YOLO model '{YOLO_MODEL_PATH}' not loaded.")
    if not os.path.exists(input_json):
        raise FileNotFoundError(f"Input JSON '{input_json}' not found.")
    with open(input_json, "r") as f:
        input_data = json.load(f)
    persona = input_data["persona"]["role"]
    job = input_data["job_to_be_done"]["task"]
    persona_query = f"{persona}. Task: {job}"
    all_section_data = []
    PDF_FOLDER = "/app/input"  # Updated for Docker
    for doc in input_data["documents"]:
        pdf_path = os.path.join(PDF_FOLDER, doc["filename"])
        if not os.path.exists(pdf_path):
            print(f"PDF '{pdf_path}' not found. Skipping.")
            continue
        print(f"Processing {pdf_path}...")
        section_data = process_pdf_sections(pdf_path, yolo_model)
        all_section_data.extend(section_data)
    if not all_section_data:
        raise RuntimeError("No sections extracted from any PDF.")
    top_sections = rank_sections(all_section_data, persona_query)
    output_json = build_output(input_data, top_sections)
    if output_file:
        with open(output_file, "w") as f:
            f.write(output_json)
        print(f"Output written to {output_file}")
    return output_json

def main():
    parser = argparse.ArgumentParser(description="Document Parsing Pipeline")
    parser.add_argument("--input_json", type=str, default="/app/input/challenge1b_input.json", help="Path to input JSON file")
    parser.add_argument("--output", type=str, default="/app/output/output.json", help="Path to output JSON file")
    args = parser.parse_args()
    try:
        parse_documents(args.input_json, args.output)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main() 