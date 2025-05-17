# Image-Based Caption Enrichment

### CS517: Major Project  
**Topic:** Image-Based Caption Enrichment  
**Submitted by:** Pankaj Singh (2021CEB1026), Rhitvik Anand (2021CEB1127), Gopal Bansal (2021CSB1089)
**Submitted to:** Prof. Puneet Goyal

---

## Overview

This project tackles the challenge of generating **richer and more informative captions** for images by integrating visual content with contextual data sourced from external textual databases.

Traditional image captioning models primarily focus on describing visible objects and actions. However, in many real-world applications—such as journalism, education, or digital archiving—understanding the **deeper context** of an image is crucial. This is where **Image-Based Caption Enrichment** comes in.

---

## Objective

Given an image, the goal is to:
- Go **beyond basic visual description**
- Search relevant articles from an **external article database**
- Extract and incorporate **contextual information**, such as:
  - Named entities (people, places, organizations)
  - Temporal and spatial cues
  - Event background and outcomes
  - Implicit relationships and significance

The resulting enriched caption should form a **coherent and comprehensive narrative** that not only describes what is seen but also communicates **why it matters**.

---

## Key Features

- 🔍 **Retrieval-Augmented Generation:** Integrates image analysis with text retrieval to enrich captions with real-world information.
- 🧠 **Context-Aware Captioning:** Captures temporal, spatial, and causal details not visible in the image.
- 🖼️ **Beyond Description:** Aims to generate captions with depth, storytelling elements, and relevance to real-world events.
- 📚 **External Knowledge Integration:** Uses a curated article database to provide supplementary context.

---

## Motivation

Standard image captions often miss critical elements like:
- **Who** is in the image (names, roles)
- **When and where** the image was taken
- **What happened** before or after the captured moment
- **Why** the moment is important

This project aims to **bridge the gap between visual data and narrative understanding**, empowering applications in areas like automated journalism, digital curation, and assistive technologies.

---

## Workflow

1. **Dataset Explanation**  
   ⇩  
2. **Generation of Embeddings of image**  
   ⇩  
3. **Summarization and Generation of Embeddings of Text**  
   ⇩  
4. **Training a Classification Model and explanation of loss fxn**  
   ⇩  
5. **Get top-k articles**  
   ⇩  
6. **Get enriched Caption**
   ⇩  
8. **Calculate metric - Recall@k , CLIP Score**

---

## Applications

- 📰 **News Automation**
- 🧾 **Documentary and Archival Tagging**
- 🎓 **Educational Tools**
- 🧑‍🦯 **Assistive Tech for Visually Impaired**

---

## Acknowledgements

This project is submitted as part of the **CS517** course at IIT Ropar, under the guidance of **Prof. Puneet Goyal**.

---

