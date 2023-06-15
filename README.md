# Negotiation
Please proceed to the `negotiation` folder.

You're a sports star and are negotiating your salary next year. Unfortunately, not everyone can join this table in person, so we decide to do this negotiation over the line. To read the salary amount (i.e., handwritten digit numbers), we will use an OCR device that runs an ML model for recognizing the handwritten digits. In the table, the team is not willing to pay more than $3 million / year for your salary next year.

Now your job is to craft an adversarial handwritten `3` that will be recognized by the OCR device as `9` so that your counter-offer does look like $`3` million, but it's actually recorded as $`9` million. To begin with, we prepared `3.png`. Your job is to craft a modified version of `3`(`3_modified.pt`). You can fix the `template.py` and run, and this will generate `3_modified.pt`.

Once you're ready, run `launcher`. If you're lucky, you will have the flag.

Good luck
