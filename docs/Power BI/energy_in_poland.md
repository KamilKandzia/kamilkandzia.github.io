---
layout: page
title: Energy and Heat
permalink: /energy_and_heat/
parent: Power BI
---

# Energy and Heat production in Poland
That's the first project that I published on my GitHub pages. For this project, I have chosen some data available online, so I found it on the open data webpage https://dane.gov.pl/. 

[Heat production 2015-2018](https://dane.gov.pl/pl/dataset/607/resource/28374/table?page=1&per_page=20&q=&sort=){: target="_blank" .btn .btn-purple }
[Energy production 2015-2018](https://dane.gov.pl/pl/dataset/607/resource/28373/table?page=1&per_page=20&q=&sort=){: target="_blank" .btn .btn-purple }

The data cover the heat and energy production in Poland by 2015-2018 divided by the voivodeship. In data investigation, I discovered that some values are missing, but it often occurs in projects. So, how to deal with it?

![Dataset preprocessing]({{site.url}}/assets/images/power_bi_files/power_bi_energy_processing.gif)

If the are missing values numerical, the values could be extrapolated by the median or mean value, but it influences the summed values in the production of energy. I decided to eliminate the voivodeships that have corrupted values. On the page "Energia cieplna" (eng. "Heat energy"), you might see missing marked voivodeships resulted from the removal of the data. In the recorded gif, I showed the steps that I have to do to preprocess the dataset to further analysis and visualization. 

After preprocessing, I created some pages that could be used in analysis values in different ways. I tried to ask myself, and I found some questions:
* Which voivodeship has the highest production of renewable energy? I created two pie charts that show the production of renewable energy in the whole country. The highlighted ring (after clicking the chosen voivodeship) indicated that condition and in a simple way presents the ratio of renewable energy.

![Renewable energy]({{site.url}}/assets/images/power_bi_files/oze_in_one_voivodeship.png)

* How the energy production change year by year? To find out, I visualize in the line chart the production per year. 

![Energy production year by year]({{site.url}}/assets/images/power_bi_files/production_energy_year_by_year.png)

* Found the voivodeships that have the highest industrialisation region in Poland. "Elektrownie i elektrociepłownie zawodowe" (eng. "Combined heat and power industry plants") present sorted the voivodeships with the highest production in the region. In the upper-silesia (pl. Śląskie), the are many resource-intensive factors in energy or heat (eg. steelworks, rolling mills, or defense industry).

![Heat energy]({{site.url}}/assets/images/power_bi_files/heat_energy_chart.png)

The final report is presented on the gif.

![Final gif power BI energy]({{site.url}}/assets/images/power_bi_files/power_bi_energy.gif)