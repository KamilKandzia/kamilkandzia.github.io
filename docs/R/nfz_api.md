---
layout: page
title: NFZ API
permalink: /nfz_api/
parent: R
---

# NFZ API
During my master's thesis, which was covered by the diabetic's topic, I have to get some data connected to that. I discovered that there were available public data of the reimbursement by the NFZ (that's the polish analog of the NHS). 

The website allows for comprehensive data analysis. In particular, the emphasis was placed on sales for given branches of the NFZ. Additionally, a year-to-year analysis was introduced. As well as analysis of the division of age groups according to the attributes.

[Live demo](https://kamil-kandzia.shinyapps.io/nfz_api/){: target="_blank" .btn .btn-purple }
![Demo]({{site.url}}/assets/images/nfz_files/nfz.gif)

On the left sidebar, there are links to the relevant tabs. On the right side, there is a product search engine by active substance or product name. By default, the first item in the Suggestion field (this is the resulting product or active substance within the dictionary method).

The choice of dates applies only for 2017 and 2018, as only such data are delivered by the government services. The NFZ division field is used in the 'One division NFZ' tab.

The additional attribute Dose allows selecting only products with a specific dose.

For more information on the NFZ API go to https://api.nfz.gov.pl/app-stat-api-ra/

## Analysis
### One division of NFZ
Analysis of one NFZ branch is available for many products. Within this functionality, two charts are displayed. Each of them concerning the sale of specific products in a given area. The first one takes the label with the dose, while the second one shows the aggregated data.

Just choose from the right sidebar NFZ division and click the appropriate button on the tab to make an analysis.

In some cases, there may appear many products with the same pill per pack and dose amount, but they differ with the EAN code.

### All divisions of NFZ
In this tab, you can analyze sales by NFZ division. Just choose from the right sidebar NFZ division and click the appropriate button.

### Yearly provision
Among the available functionalities of the NFZ API, it is possible to compare year-to-year sales of a product based on active substance or product name. Due to the availability of 2017-2018, it is only possible to compare these two years.

### Age group
Grouping is based on eight age groups available in the NFZ API (the last attribute is eliminated because it means undefined age).
