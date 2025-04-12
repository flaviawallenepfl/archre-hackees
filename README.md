# ARCHRE

# AI-Powered Writeups for Next-Gen Underwriting

## Useful Article to Understand Reinsurance

[Types of Reinsurance](https://www.munichre.com/content/dam/munichre/contentlounge/website-pieces/documents/NL-Non-Proportional_30-03-2023.pdf/_jcr_content/renditions/original./NL-Non-Proportional_30-03-2023.pdf)

## Background

Insurance is a contract where an insurer provides financial protection against risks like accidents, illness, or property damage in exchange for premiums.

Reinsurance, on the other hand, is insurance for insurers. It allows insurance companies to transfer part of their risk to a reinsurer, reducing exposure to large losses and improving financial stability.

Underwriting is the process of evaluating the risks associated with providing insurance and determining an appropriate premium.

Actuaries build models to assess risk, set pricing, and predict future claims or losses. Underwriters (UWs) then use these models—along with guidelines and professional judgment—to make approval decisions, ensuring policies align with the company's risk tolerance and profitability.

Before finalizing a reinsurance deal, the reinsurer’s underwriters and actuaries typically complete a combination of the following tasks:

- Reviewing the cedant’s historical claims data.
- Analyzing industry data for the relevant line of business.
- Assessing modeling data—both industry-wide and in-house.
- Examining year-on-year (YoY) contractual changes to understand shifts in risk exposure.
- Profiling the cedant to evaluate the accuracy of their claims handling and past loss forecasts.
- Consider macroeconomic factors such as interest rates, foreign exchange (forex) fluctuations, and inflation, and how these impact the deal.

## The Current Problem

This process is highly complex, further complicated by the lack of a standardized format for submissions (the data associated with each deal). Underwriters often spend significant time on tasks such as:

- Reviewing lengthy contractual documents to assess how legal clause changes affect risk
- Evaluating macroeconomic factors when risks are tied to specific countries or currencies
- Checking modeling data for alignment with internal models and cedant assumptions
- Profiling the cedant to determine whether their underwriting and claims handling are improving
- Creating visualizations to highlight key risk changes or financial metrics that make the deal attractive or unattractive

The final output of this process is an underwriter report. This report presents an argument for why the business should underwrite an agreed amount of risk for certain premium.

## Task

Design a tool that assists underwriters creating the content and compiling the UW report. The tool should streamline the process of analyzing the data and assessing the risks to facilitate better decisions making.

The challenge will be split into two phases:

Phase 1:
Design a tool which allows the UWs to interact with the contracts and supplementary data sets (news articles, economic data, industry data) to answer questions like:

- Highlight key contractual/legal clause differences YoY in contract X?
- Given contract X what macro-economic factors present a risk / opportunity when renewing this contract?
- What regulatory legislative changes have been announced in the last year which could result in future losses being significantly different from historical losses?

Phase 2:
How can we ingest the submission files, which are typically in in excel format, to answer complicated questions about financial viability of the deal and compare Cedant claim data to industry records? Solutions should be generalisable to different excel file formats and layouts. The tool should help answer questions like:

- Do the cedant historic losses align with industry standard models?
- What is our expected return on Layer Y based on historical losses?

Bonus Points:
Are you able to find additional supplementary data which can help the UWs improve their overall assessment and subsequently the quality of the UW report.

## Data Set

- News articles from 2022-2025
- Histroical claims data for hurricanes
- Economic Data:
  - Forex
  - Inflation
  - Interest
- Submission Data:
  - Historic claims
  - Total insured value
  - Contract wording: previous year and current year
  - Contract layers

## Guidance

To give you an understanding of what an UW report consists of, we have shared an example.

The example UW report is a property catastrophe (property cat) contract which provides cover for Wind and Hail events in the Netherlands and Belgium. (submissions/netherlands/)

In addition to the UW report, you will find the submission documents and contract wording. This is the information the UW would use to determine whether they would underwrite this risk.

## Evaluation Set

Design a tool which when prompted by an UW, can generate and compile the the content for an UW report. There are two submissions which have all the necessary data, but do not have UW reports. These are:

- Florida - Perils: Hurricane, Fire, Windstorm, Flood
- Turkey - Perils: Hurricane, Fire, Landslide, Tsunami

## Areas of Focus

As a consequence of every submission being different, UW reports do not follow a defined template. However, they typically make reference to the following:

- Notable contract changes YoY and the subsequent impact to risk / exposure.
- Macro-economic factors relevant to the proposal
- Comparison of industry data to the cedant's submission data to validate alignment.
- Use relevant news articles to highlight any shifts in risk or exposure with the region and line of business.

## The Pitch:

...

## Deep Dive Slides:

...

## Judging Criteria:

Complexity & Technical sophistication: Usage of appropriate services and technologies (10%)

Design: Usability of the solution (20%)

Viability: Possibility of realizing the solution (10%)

Feasibility: Maturity level of developed solution (10%)

Creativity & Innovation: Surprise effect to the jury (30%)

Presentation: Communication of the developed solution (20%)

## Point of Contact:

Katrin, Lorenza, Paul, Gabriel, Jude, Ben, Jon, Martin, Antoine, Juan

## Prize - the winning team members will each receive:

Arch Re branded hoody.

In addition, the winning team will be invited to a lunch/dinner with senior management at Arch Re to discuss their final product and provide more information on Arch Re and the reinsurance business.
