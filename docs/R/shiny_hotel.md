---
layout: page
title: Hotel reservation
permalink: /hotel/
parent: R
---

# Hotel reservation project
This project focused on presenting the potential of creating a hotel reservation system design by R packages.

R language is in general associated with data analysis and visualization of tables or charts. The packages allow the deployment of a functional website. Also, they are used to create a dashboard for the administration panel (e.g. by the shinydashboard). Considering some disadvantages of the free shinyapps.io version, it was chosen to focus on the functionality of the reservation system, database queries, and e-mail to the user with reservation confirmation.

### Case study
* SlickR library was used to create a hotel gallery.
* ShinyWidgets, shinythemes, and shinycssloaders were used to enhance user experience. 
* RSQLite was used to communicate with the database, the e-mail system emayili, and magrittr to parse. The QR code is displaying using qrencoder and application rendering using raster, rsvg, and svglite.
* Sending mail is done by jetmail
* The site is located on the Shinyapps server where SSL communication is provided.
* Only once the reservation can be deleted

![Demo]({{site.url}}/assets/images/hotel_files/hotel.gif)

More details of the email sent by mailjet.com:

![Demo]({{site.url}}/assets/images/hotel_files/hotel_mail.png)

