---
layout: page
title: Identification of drug interactions from the summary of product characteristics
permalink: /streamlit_drugs/
parent: Python
---

<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

# Identification of drug interactions from the summary of product characteristics
The website (in polish) allows you to process the SmPC (Summary of Product Characteristics) to find interactions between the substances. Every medicine authorized in Poland has the SmPC, including the section `Interactions with other medicinal products and other forms of interaction`. Based on this passage, I have tried to extract the names of substances that interact with the product. The list of substances is taken from the Register of Medicinal Products, which contains links to the SmPC of the medicine in question and the names of the active substances in foreign (English/Latin). The whole site was deployed using the <span class="label label-green">Streamlit</span> library. <span class="label label-green">NLP processing</span> is done using the <span class="label label-green">spacy</span> library and <span class="label label-green">thefuzz</span> for comparing names of medicinal substances 

More information about the project and live version could be found hosted on streamlit cloud service:
[link](https://share.streamlit.io/kamilkandzia/streamlit_drugs/main)

