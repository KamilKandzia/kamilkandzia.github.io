---
layout: page
title: Hotel reservation
permalink: /hotel/
parent: R
---

<details open markdown="block">
  <summary>
    Table of contents
  </summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

# Hotel reservation project
This project focused on presenting the potential of creating a hotel reservation system design by R packages.

The R language is mostly associated with data analysis (especially biomedical), visualization, and statistics. Many packages allow the deployment of a functional website. Also, they are used to create a dashboard for the administration panel (e.g. by the shinydashboard). By using the free shinyapps.io version, I have created the hotel management system. It focuses on the functionality of the reservation system, database queries, and e-mail to the user with reservation confirmation.

[Live demo](https://kamil-kandzia.shinyapps.io/portfolio/){: target="_blank" .btn .btn-purple }
[Get it from the GitHub](https://github.com/KamilKandzia/hotel_shiny){: target="_blank" .btn .btn-purple }

### Case study
* SlickR library was used to create a hotel gallery.
* ShinyWidgets, shinythemes, and shinycssloaders were used to enhance user experience. 
* RSQLite was used to communicate with the database, the e-mail system emayili, and magrittr to parse. The QR code is displayed using qrencoder and application rendering using raster, rsvg, and svglite.
* Sending mail is done by jetmail.
* The site is located on the Shinyapps server, where SSL communication is provided.
* The reservation can be deleted only once.

![Demo]({{site.url}}/assets/images/hotel_files/hotel.gif)

Scan of the QR code by Santander App:

![Demo]({{site.url}}/assets/images/hotel_files/scan_qr.gif)

More details of the email sent by mailjet.com:

![Demo]({{site.url}}/assets/images/hotel_files/hotel_mail.png)

