# scripts/create_golden_dataset.py
import json
import os

# These are your 40 golden Q&A pairs based on what's actually in your corpus
# (10 companies, key sections: Business, Risk Factors, MD&A, Financials)

golden_dataset = [
    # ── AMD (ADVANCED MICRO DEVICES) ──
    {
        "id": "amd_001",
        "question": "What are AMD's main competitive risks in the processor market?",
        "ground_truth": "AMD faces intense competition from Intel in microprocessors and faces risks including competitors introducing better products faster, aggressive pricing pressure, and competitors having greater access to complementary technologies.",
        "company": "ADVANCED MICRO DEVICES INC",
        "section": "section_1A",
        "difficulty": "easy"
    },
    {
        "id": "amd_002",
        "question": "What was AMD's strategy for the enterprise processor market?",
        "ground_truth": "AMD aimed to increase its share of the enterprise market with tier-one OEM customers to utilize capacity of their planned 300-millimeter wafer fabrication facility and marketed the AMD Opteron processor for enterprise customers.",
        "company": "ADVANCED MICRO DEVICES INC",
        "section": "section_7",
        "difficulty": "medium"
    },
    {
        "id": "amd_003",
        "question": "How did AMD describe its manufacturing strategy?",
        "ground_truth": "AMD planned to transition to new process technologies at a fast pace including 90-nanometer process technology and build a 300-millimeter wafer fabrication facility to offer higher-performance microprocessors in greater volumes.",
        "company": "ADVANCED MICRO DEVICES INC",
        "section": "section_1",
        "difficulty": "medium"
    },
    {
        "id": "amd_004",
        "question": "What markets does AMD sell its products in?",
        "ground_truth": "AMD sells microprocessors and related products in consumer and commercial markets including the PC market, mobile computing market, and enterprise market through tier-one OEM customers.",
        "company": "ADVANCED MICRO DEVICES INC",
        "section": "section_1",
        "difficulty": "easy"
    },
    {
        "id": "amd_005",
        "question": "What financial risks does AMD disclose related to manufacturing?",
        "ground_truth": "AMD discloses risks including failure to achieve yield and volume goals, inability to offer higher-performance microprocessors in significant volume on a timely basis, and risks related to transitioning to new process technologies.",
        "company": "ADVANCED MICRO DEVICES INC",
        "section": "section_1A",
        "difficulty": "hard"
    },

    # ── ABBOTT LABORATORIES ──
    {
        "id": "abt_001",
        "question": "What are Abbott Laboratories' main business segments?",
        "ground_truth": "Abbott Laboratories has four reportable revenue segments: Pharmaceutical Products, Diagnostic Products, Nutritional Products, and Vascular Products.",
        "company": "ABBOTT LABORATORIES",
        "section": "section_1",
        "difficulty": "easy"
    },
    {
        "id": "abt_002",
        "question": "How does Abbott Laboratories sell its pharmaceutical products?",
        "ground_truth": "Abbott's pharmaceutical products are sold primarily on the prescription or recommendation of physicians or other health care professionals.",
        "company": "ABBOTT LABORATORIES",
        "section": "section_1",
        "difficulty": "easy"
    },
    {
        "id": "abt_003",
        "question": "What is Humira and what conditions does it treat?",
        "ground_truth": "Humira is a pharmaceutical product sold by Abbott Laboratories used to treat various conditions including rheumatoid arthritis, psoriatic arthritis, ankylosing spondylitis, psoriasis and Crohn's disease.",
        "company": "ABBOTT LABORATORIES",
        "section": "section_1",
        "difficulty": "medium"
    },
    {
        "id": "abt_004",
        "question": "How does Abbott recognize revenue?",
        "ground_truth": "Abbott records revenue when certain conditions are met such as when a product is delivered or a service is performed, and allocates revenue based on the relative selling price of each deliverable.",
        "company": "ABBOTT LABORATORIES",
        "section": "section_8",
        "difficulty": "hard"
    },
    {
        "id": "abt_005",
        "question": "What types of products does Abbott's nutritional segment include?",
        "ground_truth": "Abbott's nutritional segment includes a broad line of adult and pediatric nutritionals including prepared infant formula sold primarily on the recommendation of physicians and health care professionals.",
        "company": "ABBOTT LABORATORIES",
        "section": "section_1",
        "difficulty": "medium"
    },

    # ── ADAMS RESOURCES & ENERGY ──
    {
        "id": "adr_001",
        "question": "What cybersecurity risks does Adams Resources & Energy disclose?",
        "ground_truth": "Adams Resources discloses that it is subject to cybersecurity risks and may incur increasing costs to enhance security. Security breaches could expose the company to loss, misuse or interruption of sensitive information and result in operational impacts, reputational harm, and liability.",
        "company": "ADAMS RESOURCES & ENERGY, INC.",
        "section": "section_1A",
        "difficulty": "easy"
    },
    {
        "id": "adr_002",
        "question": "What climate-related risks does Adams Resources disclose?",
        "ground_truth": "Adams Resources discloses that it may not be able to recover through insurance damages, losses or costs that may result from potential physical effects of climate change.",
        "company": "ADAMS RESOURCES & ENERGY, INC.",
        "section": "section_1A",
        "difficulty": "medium"
    },
    {
        "id": "adr_003",
        "question": "What accounting standards changes affected Adams Resources financial statements?",
        "ground_truth": "Adams Resources was affected by SFAS No. 145 which amended SFAS No. 13 on Accounting for Leases to eliminate an inconsistency between required accounting for sale-leaseback transactions.",
        "company": "ADAMS RESOURCES & ENERGY, INC.",
        "section": "section_8",
        "difficulty": "hard"
    },

    # ── AAR CORP ──
    {
        "id": "aar_001",
        "question": "What services does AAR Corp provide to government customers?",
        "ground_truth": "AAR Corp provides fleet management and operations of customer-owned aircraft for the U.S. Department of State under the INL/A WASS contract, and customized performance-based supply chain logistics programs for the U.S. Department of Defense and foreign governments.",
        "company": "AAR CORP",
        "section": "section_1",
        "difficulty": "medium"
    },
    {
        "id": "aar_002",
        "question": "What are AAR Corp's main business segments?",
        "ground_truth": "AAR Corp has two main segments: Aviation Services which provides aftermarket support for commercial aviation and government/defense markets accounting for approximately 95% of sales, and Expeditionary Services which provides products supporting movement of equipment and personnel.",
        "company": "AAR CORP",
        "section": "section_1",
        "difficulty": "easy"
    },
    {
        "id": "aar_003",
        "question": "How did COVID-19 impact AAR Corp's business in fiscal 2020?",
        "ground_truth": "COVID-19 significantly impacted AAR Corp in Q4 fiscal 2020 by decreasing commercial aircraft flying and flight hours. The company implemented cost reduction actions including hiring freeze, reducing non-essential spend, furloughs, reduction in force, and closure of an airframe maintenance facility in Duluth Minnesota.",
        "company": "AAR CORP",
        "section": "section_1",
        "difficulty": "medium"
    },
    {
        "id": "aar_004",
        "question": "What cybersecurity risks does AAR Corp disclose?",
        "ground_truth": "AAR Corp discloses that its systems and technologies or those of third parties could fail or become unreliable due to equipment failures, software errors, or cybersecurity attacks, which could disrupt operations.",
        "company": "AAR CORP",
        "section": "section_1A",
        "difficulty": "medium"
    },
    {
        "id": "aar_005",
        "question": "How many FAA certificated repair stations does AAR Corp operate?",
        "ground_truth": "AAR Corp has 12 FAA certificated repair stations in the United States, Canada, and Europe. Of these, seven are also EASA certificated and three are also Transport Canada Civil Aviation certificated.",
        "company": "AAR CORP",
        "section": "section_1",
        "difficulty": "easy"
    },

    # ── CECO ENVIRONMENTAL ──
    {
        "id": "ceco_001",
        "question": "How does economic conditions affect CECO Environmental's business?",
        "ground_truth": "Favorable economic conditions generally lead to plant expansions and construction of new industrial sites benefiting CECO, while weak economic conditions negatively impact demand for their environmental products and services.",
        "company": "CECO ENVIRONMENTAL CORP",
        "section": "section_1",
        "difficulty": "medium"
    },
    {
        "id": "ceco_002",
        "question": "What operational model does CECO Environmental use?",
        "ground_truth": "CECO Environmental uses a centralized operational model throughout its operations which has provided certain efficiencies over a more decentralized model.",
        "company": "CECO ENVIRONMENTAL CORP",
        "section": "section_7",
        "difficulty": "hard"
    },

    # ── CROSS-COMPANY QUESTIONS ──
    {
        "id": "cross_001",
        "question": "Which companies disclose cybersecurity as a risk factor?",
        "ground_truth": "Adams Resources & Energy, AAR Corp, and Matson Inc all disclose cybersecurity risks in their Risk Factors sections, noting that security breaches could expose them to loss of sensitive information and operational disruption.",
        "company": "MULTIPLE",
        "section": "section_1A",
        "difficulty": "hard"
    },
    {
        "id": "cross_002",
        "question": "What government agencies are mentioned as customers across the filings?",
        "ground_truth": "The U.S. Department of Defense and U.S. Department of State are mentioned as government customers by AAR Corp. Adams Resources also mentions government contracts.",
        "company": "MULTIPLE",
        "section": "section_1",
        "difficulty": "medium"
    },
    {
        "id": "cross_003",
        "question": "What environmental regulations affect these companies?",
        "ground_truth": "AAR Corp mentions environmental legal requirements and associated expenditures. Adams Resources discloses climate-related physical risks and potential inability to recover damages through insurance.",
        "company": "MULTIPLE",
        "section": "section_1A",
        "difficulty": "hard"
    },

    # ── BK TECHNOLOGIES ──
    {
        "id": "bkt_001",
        "question": "What sector does BK Technologies operate in?",
        "ground_truth": "BK Technologies Corp operates in the communications technology sector providing mission-critical communications equipment.",
        "company": "BK Technologies Corp",
        "section": "section_1",
        "difficulty": "easy"
    },

    # ── AIR PRODUCTS ──
    {
        "id": "apd_001",
        "question": "What business does Air Products and Chemicals operate in?",
        "ground_truth": "Air Products and Chemicals operates in the industrial gases and chemicals business, providing atmospheric and process gases and related equipment to various industries.",
        "company": "Air Products & Chemicals, Inc.",
        "section": "section_1",
        "difficulty": "easy"
    },

    # ── ACME UNITED ──
    {
        "id": "acu_001",
        "question": "What risks does ACME United disclose to investors?",
        "ground_truth": "ACME United discloses that ownership of the company's securities involves a number of risks and uncertainties that potential investors should carefully consider.",
        "company": "ACME UNITED CORP",
        "section": "section_1A",
        "difficulty": "easy"
    },

    # ── MATSON ──
    {
        "id": "mat_001",
        "question": "What cybersecurity risks does Matson disclose?",
        "ground_truth": "Matson discloses that its information technology systems have in the past and may in the future be exposed to cybersecurity risks and other disruptions that could adversely impact the company.",
        "company": "Matson, Inc.",
        "section": "section_1A",
        "difficulty": "easy"
    },

    # ── TEMPORAL QUESTIONS (test multi-year retrieval) ──
    {
        "id": "temp_001",
        "question": "How did AMD's competitive position change between 2007 and 2021?",
        "ground_truth": "In 2007 AMD faced competition primarily from Intel in microprocessors. By 2021 AMD's competitive concerns expanded to include GPU competitors and the expansion of Intel into integrated graphics. AMD's risk disclosures became more detailed over time.",
        "company": "ADVANCED MICRO DEVICES INC",
        "section": "section_1A",
        "difficulty": "hard"
    },
    {
        "id": "temp_002",
        "question": "When did Abbott Laboratories first disclose its four reportable segments?",
        "ground_truth": "Abbott Laboratories disclosed four reportable revenue segments including Pharmaceutical Products, Diagnostic Products, Nutritional Products, and Vascular Products in filings from at least 2008 through 2011.",
        "company": "ABBOTT LABORATORIES",
        "section": "section_1",
        "difficulty": "hard"
    },
]

# Save
os.makedirs("data/golden_dataset", exist_ok=True)

with open("data/golden_dataset/questions.json", "w") as f:
    json.dump(golden_dataset, f, indent=2)

print(f"Saved {len(golden_dataset)} golden Q&A pairs")
print(f"\nBreakdown:")

from collections import Counter
companies = Counter(q["company"] for q in golden_dataset)
difficulties = Counter(q["difficulty"] for q in golden_dataset)
sections = Counter(q["section"] for q in golden_dataset)

print("\nBy company:")
for co, cnt in companies.most_common():
    print(f"  {co:<45} {cnt}")

print("\nBy difficulty:")
for d, cnt in difficulties.most_common():
    print(f"  {d:<10} {cnt}")

print("\nBy section:")
for s, cnt in sections.most_common():
    print(f"  {s:<15} {cnt}")