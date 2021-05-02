---
layout: page
title: Dataset visualizations
permalink: /datasets/
parent: R
---

# Dataset analysis
During my master thesis, I created a website to get data from by the NFZ API. I enjoyed the final site visual effects implementation on R by the Shiny apps. Nevertheless, why not deploy a new website with more fancy charts?

That was the idea which I had in my mind by creating another page on Shiny.

## Case study
I started by finding a new dataset that I can use to analyse, and in the next stage, visualize it. I choose the "Accidental Drug-Related Deaths" dataset. The dataset contains death associated with drug overdose in Connecticut from 2012 to 2018. Created by the Office of the Chief Medical Examiner, data includes the toxicity report, death certificate, and scene investigation.

I have done some of the data cleanings, but the main aim of the analysis was to create fancy and colorful charts on the Shiny apps. I chose ECharts2Shiny, but I realized that the library contains a Chinese sign in the function, so my RStudio cannot correctly use this library. I had to fork and modify one of the functions https://github.com/KamilKandzia/ECharts2Shiny. 

![Datasets]({{site.url}}/assets/images/datasets_files/datasets.gif)