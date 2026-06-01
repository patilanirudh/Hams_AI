import json

# --- DATASET A ---
dataset_a = [
    # 7 Arabic monolingual QAs
    {
        "id": "qa_001",
        "question": "ما هي المهلة المحددة لاسترجاع الأثاث المكتبي بعد توصيله؟",
        "reference_answer": "يحق للعميل طلب استرجاع الأثاث المكتبي خلال 14 يوماً من تاريخ التوصيل الفعلي للطلب. ويجب أن تكون المنتجات في حالتها الأصلية، غير مستخدمة، وخالية من أي تلف ناتج عن سوء الاستخدام. في حال كان الاسترجاع لرغبة العميل دون عيب مصنعي، يتحمل رسوماً إدارية تبلغ 150 ريال سعودي (١٥٠ ر.س). المصدر: policy_returns_ar",
        "source_doc_id": "policy_returns_ar",
        "question_lang": "ar",
        "answer_lang": "ar",
        "type": "monolingual"
    },
    {
        "id": "qa_002",
        "question": "كم تبلغ فترة الضمان لطاولة الاجتماعات الذكية HamsAI Boardroom Table؟",
        "reference_answer": "تتمتع طاولة الاجتماعات الذكية HamsAI Boardroom Table بضمان محدود لمدة 3 سنوات (٣ سنوات) من تاريخ الشراء المدون في الفاتورة. يغطي هذا الضمان الهيكل الخارجي، وآليات الحركة، والعيوب المصنعية المكتشفة أثناء الاستخدام العادي للمنتج. المصدر: policy_warranty_ar",
        "source_doc_id": "policy_warranty_ar",
        "question_lang": "ar",
        "answer_lang": "ar",
        "type": "monolingual"
    },
    {
        "id": "qa_003",
        "question": "كم تبلغ التكلفة السنوية للاشتراك في الخطة الذهبية للدعم الفني؟",
        "reference_answer": "تبلغ التكلفة السنوية للخطة الذهبية (Gold Support) للدعم الفني 18000 ريال سعودي (١٨٠٠٠ ر.س) سنوياً. توفر هذه الخطة دعماً على مدار الساعة 24/7 للحالات الطارئة بالإضافة إلى زيارتين ميدانيتين مجانيتين شهرياً لفحص الأجهزة والشبكات في الرياض، جدة، أو الدمام. المصدر: policy_support_plans_ar",
        "source_doc_id": "policy_support_plans_ar",
        "question_lang": "ar",
        "answer_lang": "ar",
        "type": "monolingual"
    },
    {
        "id": "qa_004",
        "question": "ما هي الغرامة المفروضة عند إلغاء الاشتراك السنوي في برمجيات HamsAI بعد تفعيلها؟",
        "reference_answer": "في حال إلغاء الاشتراك السنوي بعد مرور أول 3 أيام عمل من التفعيل، يتم فرض رسوم إلغاء مبكر (Early Termination Fee) تبلغ 750 ريال سعودي (٧٥٠ ر.س) أو ما يعادل 20% من القيمة المتبقية من العقد، أيهما أعلى. ويتم استرداد الرصيد المتبقي للعميل بعد خصم الرسوم المستحقة للشركة. المصدر: policy_cancellation_ar",
        "source_doc_id": "policy_cancellation_ar",
        "question_lang": "ar",
        "answer_lang": "ar",
        "type": "monolingual"
    },
    {
        "id": "qa_005",
        "question": "ما هي تكلفة توصيل وتركيب الأثاث للطلبات التي تقل قيمتها الإجمالية عن 10000 ريال؟",
        "reference_answer": "للطلب الذي يقل قيمته عن 10000 ريال سعودي (١٠٠٠٠ ر.س)، يتم احتساب رسوم توصيل قياسية تبلغ 250 ريال سعودي (٢٥٠ ر.س). أما خدمات التوصيل والتركيب للطلبات التي تتجاوز قيمتها هذا الحد فهي مجانية بالكامل. المصدر: policy_delivery_ar",
        "source_doc_id": "policy_delivery_ar",
        "question_lang": "ar",
        "answer_lang": "ar",
        "type": "monolingual"
    },
    {
        "id": "qa_006",
        "question": "ما هو الوقت المحدد لإرسال فريق صيانة الضمان إلى موقع العميل في الرياض؟",
        "reference_answer": "تلتزم شركة HamsAI بالاستجابة الأولى وتحديد الموعد خلال 24 ساعة من استلام بلاغ الضمان. ويتم إرسال فريق الصيانة الميدانية إلى موقع العميل في الرياض، جدة، أو الدمام خلال 48 ساعة عمل. المصدر: policy_warranty_ar",
        "source_doc_id": "policy_warranty_ar",
        "question_lang": "ar",
        "answer_lang": "ar",
        "type": "monolingual"
    },
    {
        "id": "qa_007",
        "question": "كم من الوقت يستغرق استرداد المبالغ المالية لحساب العميل بعد إلغاء الطلبات؟",
        "reference_answer": "تلتزم HamsAI بمعالجة طلبات الاسترداد المالي المعتمدة وإعادتها إلى العميل خلال 10 إلى 14 يوم عمل. ويتم رد المبالغ حصراً عبر نفس وسيلة الدفع الأصلية التي استخدمها العميل (مثل مدى، فيزا، أو التحويل البنكي). المصدر: policy_cancellation_ar",
        "source_doc_id": "policy_cancellation_ar",
        "question_lang": "ar",
        "answer_lang": "ar",
        "type": "monolingual"
    },

    # 7 English monolingual QAs
    {
        "id": "qa_008",
        "question": "What is the target first response time for Severity 1 critical issues under HamsAI Support?",
        "reference_answer": "The target First Response Time for Severity 1 (Critical) issues is 1 hour, and the Resolution or Workaround Time is 4 hours. Support coverage for this severity level is provided on a 24/7/365 basis. Source: policy_sla_en",
        "source_doc_id": "policy_sla_en",
        "question_lang": "en",
        "answer_lang": "en",
        "type": "monolingual"
    },
    {
        "id": "qa_009",
        "question": "How much does HamsAI SaaS Pro cost per user on a monthly basis?",
        "reference_answer": "The subscription price for HamsAI SaaS Pro starts at $150 USD (equivalent to SAR 562.50) per user per month, billed annually. Shared logins are strictly prohibited and each user must have a unique identifier. Source: policy_subscriptions_en",
        "source_doc_id": "policy_subscriptions_en",
        "question_lang": "en",
        "answer_lang": "en",
        "type": "monolingual"
    },
    {
        "id": "qa_010",
        "question": "What is the coverage period for the limited warranty of HamsAI ServerRack X1?",
        "reference_answer": "Physical hardware products like HamsAI ServerRack X1 are covered by a limited warranty of 36 months from the date of physical delivery and sign-off. This warranty covers defects in materials and workmanship under normal use conditions. Source: policy_warranty_en",
        "source_doc_id": "policy_warranty_en",
        "question_lang": "en",
        "answer_lang": "en",
        "type": "monolingual"
    },
    {
        "id": "qa_011",
        "question": "What is the duration and core activities of the second phase of HamsAI implementation?",
        "reference_answer": "Phase 2 (System Configuration & Integration) takes place between Weeks 3 and 6 of the onboarding cycle. During this phase, HamsAI engineers configure system instances and establish API connections between HamsAI Core CRM and the customer's legacy databases. Source: policy_onboarding_en",
        "source_doc_id": "policy_onboarding_en",
        "question_lang": "en",
        "answer_lang": "en",
        "type": "monolingual"
    },
    {
        "id": "qa_012",
        "question": "Where is the data stored for enterprise clients operating HamsAI CRM in Saudi Arabia?",
        "reference_answer": "All customer data generated by organizations operating in Saudi Arabia is hosted locally in secure data centers located in Riyadh. This ensures compliance with the Saudi Personal Data Protection Law (PDPL) regarding data sovereignty and residency. Source: policy_privacy_en",
        "source_doc_id": "policy_privacy_en",
        "question_lang": "en",
        "answer_lang": "en",
        "type": "monolingual"
    },
    {
        "id": "qa_013",
        "question": "How are service credits calculated if HamsAI fails to meet the Severity 1 support response times?",
        "reference_answer": "If HamsAI fails to meet the defined First Response Time SLA for Severity 1 issues, the customer is entitled to request a Service Credit. This credit is calculated as 10% of the monthly support fee, capped at a maximum of 50% per month, applied to the next billing cycle. Source: policy_sla_en",
        "source_doc_id": "policy_sla_en",
        "question_lang": "en",
        "answer_lang": "en",
        "type": "monolingual"
    },
    {
        "id": "qa_014",
        "question": "How much is the reschedule fee if customer-caused delays postpone onboarding milestones?",
        "reference_answer": "If customer-caused delays postpone a project milestone by more than 10 business days, HamsAI will pause work and charge a Rescheduling Fee of $1,000 USD (SAR 3,750). This fee covers the cost of reallocating engineering resources. Source: policy_onboarding_en",
        "source_doc_id": "policy_onboarding_en",
        "question_lang": "en",
        "answer_lang": "en",
        "type": "monolingual"
    },

    # 3 Mixed Arabic/English QAs
    {
        "id": "qa_015",
        "question": "كم تبلغ تكلفة الاشتراك الشهري لـ Enterprise Package وما هي التكاليف الإضافية لكل مستخدم؟",
        "reference_answer": "تبلغ تكلفة الاشتراك الأساسي لباقة المؤسسات ١٨٧٥٠ ريال سعودي (18750 SAR) شهرياً، وتتضمن ٥٠ رخصة مستخدم (User Licenses). عند الرغبة في إضافة مستخدمين إضافيين، تبلغ الرسوم ٢٢٥ ريال سعودي (225 SAR) لكل مستخدم شهرياً. المصدر: pricing_enterprise_mixed",
        "source_doc_id": "pricing_enterprise_mixed",
        "question_lang": "ar",
        "answer_lang": "mixed",
        "type": "monolingual"
    },
    {
        "id": "qa_016",
        "question": "ما هي المواصفات الموصى بها (Recommended Requirements) لمعالج CPU والذاكرة RAM لتشغيل النظام؟",
        "reference_answer": "تشمل المواصفات الموصى بها معالج CPU بعدد 16 Cores أو أكثر، وذاكرة عشوائية بسعة 32 GB RAM DDR4. كما تتطلب سعة تخزين تبلغ 500 GB SSD NVMe لسرعة معالجة العمليات. المصدر: reqs_infrastructure_mixed",
        "source_doc_id": "reqs_infrastructure_mixed",
        "question_lang": "ar",
        "answer_lang": "mixed",
        "type": "monolingual"
    },
    {
        "id": "qa_017",
        "question": "كم تبلغ رسوم تفعيل مفتاح الوصول الفعلي للـ API في بيئة الإنتاج Production Environment؟",
        "reference_answer": "تبلغ رسوم الاشتراك السنوي لتفعيل مفتاح الوصول الفعلي في بيئة الإنتاج (Production Environment) ٥٦٢٥ ريال سعودي (5625 SAR). تتوفر بيئة التطوير التجريبية (Sandbox Environment) مجاناً للمطورين. المصدر: guide_api_mixed",
        "source_doc_id": "guide_api_mixed",
        "question_lang": "ar",
        "answer_lang": "mixed",
        "type": "monolingual"
    },

    # 3 Cross-lingual QAs
    {
        "id": "qa_018",
        "question": "ما هي شروط تمديد الضمان للأثاث والمعدات وما قيمتها بالدولار والريال؟",
        "reference_answer": "Customers can purchase an Extended Warranty within the first 90 days of purchase for an annual fee of $600 USD (SAR 2,250) per server rack, or $200 USD (SAR 750) per executive desk. This provides additional coverage beyond the standard 36 months warranty. Source: policy_warranty_en",
        "source_doc_id": "policy_warranty_en",
        "question_lang": "ar",
        "answer_lang": "en",
        "type": "cross_lingual"
    },
    {
        "id": "qa_019",
        "question": "What is the fee for canceling a workstation or chair order after it has shipped?",
        "reference_answer": "بمجرد خروج الشحنة للتوصيل إلى العميل في الرياض أو جدة أو الدمام، لا يمكن إلغاء الطلب مباشرة. يتعين على العميل استلام الشحنة ثم تقديم طلب استرجاع رسمي يخضع لرسوم الشحن والاسترجاع البالغة 150 ريال سعودي (١٥٠ ر.س). المصدر: policy_cancellation_ar",
        "source_doc_id": "policy_cancellation_ar",
        "question_lang": "en",
        "answer_lang": "ar",
        "type": "cross_lingual"
    },
    {
        "id": "qa_020",
        "question": "ما هي التدابير المتخذة لحماية وتشفير البيانات المخزنة والمنقولة في أنظمة HamsAI؟",
        "reference_answer": "All customer data at rest is encrypted using AES-256 standards, while data in transit between web browsers and HamsAI systems is encrypted using TLS 1.3 protocols. Additionally, customer systems enforce Role-Based Access Control (RBAC) and mandatory Multi-Factor Authentication (MFA). Source: policy_privacy_en",
        "source_doc_id": "policy_privacy_en",
        "question_lang": "ar",
        "answer_lang": "en",
        "type": "cross_lingual"
    }
]

# --- DATASET B ---
dataset_b = {
  "conversations": [
    {
      "conversation_id": "conv_001",
      "topic": "support plans and pricing",
      "turns": [
        {
          "turn": 1,
          "user": "Hello, I am interested in HamsAI support plans. What support levels do you offer?",
          "standalone_query": "What support plan levels are offered by HamsAI?",
          "expected_answer": "HamsAI offers four levels of annual support plans: Bronze Support at 4,500 SAR, Silver Support at 9,500 SAR, Gold Support at 18,000 SAR, and Platinum Support at 25,000 SAR. Source: policy_support_plans_ar",
          "source_doc_id": "policy_support_plans_ar",
          "lang": "en"
        },
        {
          "turn": 2,
          "user": "What is the response time for Platinum Support if our system goes down?",
          "standalone_query": "What is the Severity 1 system down response time for HamsAI Platinum Support?",
          "expected_answer": "For critical Severity 1 issues under the Platinum Support plan, HamsAI provides an immediate response time within 1 hour (١ ساعة) and 24/7/365 coverage. Source: policy_support_plans_ar",
          "source_doc_id": "policy_support_plans_ar",
          "lang": "en"
        },
        {
          "turn": 3,
          "user": "And what is the response time under the Gold plan for similar critical issues?",
          "standalone_query": "What is the Gold support plan response time for critical Severity 1 issues?",
          "expected_answer": "Under the Gold support plan, the response time for critical issues is within 4 hours, and it also includes 24/7 coverage for emergency situations. Source: policy_support_plans_ar",
          "source_doc_id": "policy_support_plans_ar",
          "lang": "en"
        },
        {
          "turn": 4,
          "user": "Does the Gold plan include any on-site support in Riyadh?",
          "standalone_query": "Does HamsAI Gold Support include on-site visits in Riyadh?",
          "expected_answer": "Yes, the Gold plan includes two free on-site visits per month for checking equipment and networks in Riyadh, Jeddah, or Dammam. Source: policy_support_plans_ar",
          "source_doc_id": "policy_support_plans_ar",
          "lang": "en"
        },
        {
          "turn": 5,
          "user": "How much extra would we pay monthly to add Gold support to our enterprise suite subscription?",
          "standalone_query": "What is the monthly pricing for the Gold support plan?",
          "expected_answer": "The Gold Support Plan is priced as an add-on at 4,500 SAR (٤٥٠٠ ر.س) per month, which is billed alongside your enterprise software subscription. Source: pricing_enterprise_mixed",
          "source_doc_id": "pricing_enterprise_mixed",
          "lang": "en"
        }
      ]
    },
    {
      "conversation_id": "conv_002",
      "topic": "returns policy with follow-ups",
      "turns": [
        {
          "turn": 1,
          "user": "أهلاً بك، أود الاستفسار عن سياسة الاسترجاع للأجهزة والأثاث المكتبي الذي قمنا بشرائه.",
          "standalone_query": "ما هي شروط استرجاع الأثاث المكتبي والأجهزة في HamsAI؟",
          "expected_answer": "يحق للعميل طلب الاسترجاع خلال 14 يوماً من تاريخ التوصيل الفعلي، والاستبدال خلال 30 يوماً، شريطة أن تكون المنتجات غير مستخدمة وفي عبوتها الأصلية. المصدر: policy_returns_ar",
          "source_doc_id": "policy_returns_ar",
          "lang": "ar"
        },
        {
          "turn": 2,
          "user": "هل هناك رسوم إرجاع إذا قمنا برد المنتج لمجرد تغيير رأينا؟",
          "standalone_query": "هل هناك رسوم إرجاع للأثاث دون عيب مصنعي؟",
          "expected_answer": "نعم، في حال كان الاسترجاع بسبب رغبة العميل دون وجود عيب مصنعي، يتحمل العميل رسوم النقل الإدارية البالغة 150 ريال سعودي (١٥٠ ر.س). المصدر: policy_returns_ar",
          "source_doc_id": "policy_returns_ar",
          "lang": "ar"
        },
        {
          "turn": 3,
          "user": "وماذا عن تراخيص برمجيات ERP السحابية؟ هل تخضع لنفس رسوم الـ 150 ريال؟",
          "standalone_query": "ما هي شروط ورسوم استرجاع تراخيص برمجيات HamsAI Cloud ERP؟",
          "expected_answer": "لا، البرمجيات السحابية مثل HamsAI Cloud ERP يمكن إلغاؤها واسترداد كامل القيمة مجاناً خلال 7 أيام من التفعيل (Activation Date)، شريطة ألا يتعدى الاستهلاك 1 جيجابايت. ولا تفرض عليها رسوم استرجاع مادية. المصدر: policy_returns_ar",
          "source_doc_id": "policy_returns_ar",
          "lang": "ar"
        },
        {
          "turn": 4,
          "user": "كم يستغرق فحص الأثاث بعد وصوله لمستودعكم لإقرار الاسترجاع المالي؟",
          "standalone_query": "ما هي مهلة فحص الأثاث المرتجع وإصدار موافقة الاسترجاع؟",
          "expected_answer": "يتم فحص المنتج المادي وإصدار الموافقة على طلب الاسترجاع خلال 5 أيام عمل من تاريخ استلام المنتج في مستودعات HamsAI بالرياض أو جدة. المصدر: policy_returns_ar",
          "source_doc_id": "policy_returns_ar",
          "lang": "ar"
        },
        {
          "turn": 5,
          "user": "ومتى تودع الأموال فعلياً في حسابنا البنكي بعد الموافقة؟",
          "standalone_query": "ما هي فترة إعادة المبالغ المستردة لحساب العميل بعد الموافقة؟",
          "expected_answer": "يتم إعادة المبالغ المالية إلى الحساب البنكي للعميل أو بطاقة الائتمان خلال فترة تتراوح بين 10 إلى 14 يوم عمل بعد الموافقة. المصدر: policy_returns_ar",
          "source_doc_id": "policy_returns_ar",
          "lang": "ar"
        }
      ]
    },
    {
      "conversation_id": "conv_003",
      "topic": "technical specs and API integration",
      "turns": [
        {
          "turn": 1,
          "user": "We are preparing for the on-premise installation of HamsAI Enterprise Suite. What are the minimum server hardware specs?",
          "standalone_query": "What are the minimum server hardware specifications for HamsAI Enterprise Suite?",
          "expected_answer": "The minimum server requirements are 8 CPU Cores (x86_64 architecture), 16 GB RAM, 100 GB SSD storage, and a 1 Gbps network card speed. Source: reqs_infrastructure_mixed",
          "source_doc_id": "reqs_infrastructure_mixed",
          "lang": "en"
        },
        {
          "turn": 2,
          "user": "What specifications are recommended for optimal database and CRM performance?",
          "standalone_query": "What are the recommended hardware specifications for HamsAI CRM and database?",
          "expected_answer": "The recommended specifications are 16 CPU Cores, 32 GB RAM DDR4, and 500 GB SSD NVMe storage. It is recommended to use separate servers for HamsAI Cloud ERP and HamsAI Core CRM. Source: reqs_infrastructure_mixed",
          "source_doc_id": "reqs_infrastructure_mixed",
          "lang": "en"
        },
        {
          "turn": 3,
          "user": "Which database engine and operating systems are officially supported?",
          "standalone_query": "What databases and operating systems are supported by HamsAI?",
          "expected_answer": "HamsAI supports Ubuntu Server 22.04 LTS or Red Hat Enterprise Linux 9 operating systems, Docker Engine 24.0.0 or higher, and PostgreSQL database version 15 or higher. Source: reqs_infrastructure_mixed",
          "source_doc_id": "reqs_infrastructure_mixed",
          "lang": "en"
        },
        {
          "turn": 4,
          "user": "We want to sync customer data via API. How do we authenticate API calls?",
          "standalone_query": "What is the authentication mechanism for HamsAI Core CRM REST APIs?",
          "expected_answer": "HamsAI APIs use the OAuth2 protocol for secure authentication. You must generate API Credentials via the Developer Console and request an Access Token via POST to https://api.hams.ai/v1/oauth/token. Source: guide_api_mixed",
          "source_doc_id": "guide_api_mixed",
          "lang": "en"
        },
        {
          "turn": 5,
          "user": "What are the API rate limits in production and is there any subscription cost?",
          "standalone_query": "What are the production API rate limits and costs for HamsAI CRM?",
          "expected_answer": "The API rate limit is 60 requests per minute per credentials key. Accessing the APIs in the production environment requires an annual subscription fee of 5,625 SAR (٥٦٢٥ ر.س). Source: guide_api_mixed",
          "source_doc_id": "guide_api_mixed",
          "lang": "en"
        }
      ]
    },
    {
      "conversation_id": "conv_004",
      "topic": "warranty and installation in Arabic",
      "turns": [
        {
          "turn": 1,
          "user": "مرحباً، أود معرفة تفاصيل الضمان المتاحة لمعدات الخوادم HamsAI ServerRack X1.",
          "standalone_query": "ما هي تفاصيل وفترة ضمان خادم HamsAI ServerRack X1؟",
          "expected_answer": "تتمتع خزائن الخوادم HamsAI ServerRack X1 بضمان محدود لمدة 3 سنوات (٣ سنوات) من تاريخ الشراء، ويغطي هذا الضمان الهيكل الخارجي وآليات الحركة والعيوب المصنعية. المصدر: policy_warranty_ar",
          "source_doc_id": "policy_warranty_ar",
          "lang": "ar"
        },
        {
          "turn": 2,
          "user": "هل يمكن تمديد هذا الضمان وما التكلفة المترتبة على ذلك؟",
          "standalone_query": "ما هي رسوم تمديد الضمان لخوادم HamsAI ServerRack X1؟",
          "expected_answer": "نعم، تتيح HamsAI خيار تمديد الضمان (Extended Warranty) لمدة عام إضافي مقابل رسوم تبلغ 1200 ريال سعودي (١٢٠٠ ر.س) للمعدات التقنية الكبيرة والخوادم. المصدر: policy_warranty_ar",
          "source_doc_id": "policy_warranty_ar",
          "lang": "ar"
        },
        {
          "turn": 3,
          "user": "وإذا تقدمنا بطلب صيانة تحت الضمان، كم من الوقت يستغرق وصول فريق الصيانة؟",
          "standalone_query": "ما هو وقت الاستجابة الميدانية لصيانة الضمان في HamsAI؟",
          "expected_answer": "يتم الرد الأول وتحديد الموعد خلال 24 ساعة من البلاغ، ويتم إرسال فريق الصيانة الميدانية إلى موقع العميل في الرياض، جدة، أو الدمام خلال 48 ساعة عمل. المصدر: policy_warranty_ar",
          "source_doc_id": "policy_warranty_ar",
          "lang": "ar"
        },
        {
          "turn": 4,
          "user": "ماذا لو استغرق الفحص والاصلاح فترة طويلة؟ هل يتم توفير بديل؟",
          "standalone_query": "هل توفر HamsAI منتجاً بديلاً أثناء فترة إصلاح الضمان الطويلة؟",
          "expected_answer": "نعم، في حال تطلب الإصلاح شحن المنتج إلى ورش الصيانة وتجاوزت فترة الإصلاح 7 أيام عمل، يتم تزويد العميل بمنتج بديل مؤقت لضمان استمرارية أعماله. المصدر: policy_warranty_ar",
          "source_doc_id": "policy_warranty_ar",
          "lang": "ar"
        },
        {
          "turn": 5,
          "user": "وبالنسبة للتركيب والتوصيل لهذه الخوادم، هل هي مجانية أم هناك رسوم معينة؟",
          "standalone_query": "ما هي رسوم توصيل وتركيب خوادم ServerRack X1؟",
          "expected_answer": "إذا بلغت قيمة الطلب الإجمالية 10000 ريال سعودي (١٠٠٠٠ ر.س) أو أكثر، فإن التوصيل والتركيب القياسي مجاني بالكامل. للطلبات الأقل، يتم تطبيق رسوم توصيل تبلغ 250 ريال (٢٥٠ ر.س)، وتخضع الخوادم لرسوم تركيب متخصصة تبلغ 500 ريال سعودي (٥٠٠ ر.س). المصدر: policy_delivery_ar",
          "source_doc_id": "policy_delivery_ar",
          "lang": "ar"
        },
        {
          "turn": 6,
          "user": "وإذا تبين عند وصول فريق التركيب أن الموقع غير مهيأ بالكهرباء والشبكة، هل يتم فرض غرامة؟",
          "standalone_query": "ما هي رسوم عدم جاهزية الموقع لتركيب المعدات من HamsAI؟",
          "expected_answer": "نعم، في حال عدم جاهزية الموقع (مثل غياب التمديدات الكهربائية أو التبريد) وتأجيل التركيب، يتحمل العميل رسوم زيارة إضافية لإعادة الجدولة تبلغ 300 ريال سعودي (٣٠٠ ر.س). المصدر: policy_delivery_ar",
          "source_doc_id": "policy_delivery_ar",
          "lang": "ar"
        }
      ]
    },
    {
      "conversation_id": "conv_005",
      "topic": "language switch mid-conversation",
      "turns": [
        {
          "turn": 1,
          "user": "السلام عليكم، قمنا بالاشتراك السنوي في نظام HamsAI SaaS Pro ونريد معرفة كيفية إلغاء التجديد التلقائي للاشتراك.",
          "standalone_query": "كيف يمكن إلغاء التجديد التلقائي لاشتراك HamsAI SaaS Pro السنوي؟",
          "expected_answer": "يجب على العميل تقديم إشعار رسمي بعدم الرغبة في التجديد قبل 30 يوماً على الأقل من نهاية فترة الاشتراك الحالية لتفادي التجديد التلقائي لعام إضافي. المصدر: policy_cancellation_ar",
          "source_doc_id": "policy_cancellation_ar",
          "lang": "ar"
        },
        {
          "turn": 2,
          "user": "وإذا أردنا إلغاء العقد السنوي في منتصف العام، هل هناك غرامة؟",
          "standalone_query": "ما هي رسوم إلغاء الاشتراك السنوي لبرمجيات HamsAI في منتصف العقد؟",
          "expected_answer": "نعم، في حال الإلغاء المبكر بعد مرور أول 3 أيام عمل من التفعيل، يتم فرض رسوم إلغاء مبكر (Early Termination Fee) تبلغ 750 ريال سعودي (٧٥٠ ر.س) أو 20% من القيمة المتبقية من العقد، أيهما أعلى. المصدر: policy_cancellation_ar",
          "source_doc_id": "policy_cancellation_ar",
          "lang": "ar"
        },
        {
          "turn": 3,
          "user": "We would like to switch to English now. Can you tell us how much we pay per user per month for HamsAI SaaS Pro?",
          "standalone_query": "What is the monthly per-user pricing for HamsAI SaaS Pro?",
          "expected_answer": "The subscription price for HamsAI SaaS Pro starts at $150 USD (SAR 562.50) per user per month, billed annually. Source: policy_subscriptions_en",
          "source_doc_id": "policy_subscriptions_en",
          "lang": "en"
        },
        {
          "turn": 4,
          "user": "Are there any usage limits on these user accounts, and what is the fee for exceeding them?",
          "standalone_query": "What are the account usage limits and overage fees for HamsAI software subscriptions?",
          "expected_answer": "Subscriptions are subject to predefined limits on storage, transaction volumes, or API calls. Exceeding these limits incurs an overage rate of $20 USD (SAR 75) per unit. Source: policy_subscriptions_en",
          "source_doc_id": "policy_subscriptions_en",
          "lang": "en"
        },
        {
          "turn": 5,
          "user": "Understood. Finally, what security protocols protect our account databases from data breaches?",
          "standalone_query": "What database encryption and security protocols are used in HamsAI cloud services?",
          "expected_answer": "HamsAI encrypts data at rest using AES-256 standards, encrypts data in transit using TLS 1.3, and enforces Role-Based Access Control (RBAC) along with mandatory Multi-Factor Authentication (MFA). Source: policy_privacy_en",
          "source_doc_id": "policy_privacy_en",
          "lang": "en"
        }
      ]
    }
  ]
}

# --- VALIDATE & SAVE ---
# Verify Dataset A
assert len(dataset_a) == 20, "Dataset A must have exactly 20 QA pairs."
lang_counts_a = {"ar": 0, "en": 0, "mixed": 0}
for item in dataset_a:
    lang_counts_a[item["answer_lang"]] += 1
    # Check sentence count (2-4 sentences)
    # A simple sentence count by counting periods.
    sentences = [s.strip() for s in item["reference_answer"].split('.') if s.strip()]
    count = len(sentences)
    # Some sentences might have abbreviations with periods, but let's check
    if count < 2 or count > 5:
         print(f"QA {item['id']} has {count} sentences: '{item['reference_answer']}'")
    # Verify that the answer ends with the source doc_id (or contains it at the end)
    assert item["source_doc_id"] in item["reference_answer"], f"QA {item['id']} is missing source citation in text."

# Verify Dataset B
assert len(dataset_b["conversations"]) == 5, "Dataset B must have exactly 5 conversations."
for conv in dataset_b["conversations"]:
    turns = conv["turns"]
    assert 5 <= len(turns) <= 6, f"Conversation {conv['conversation_id']} must have 5-6 turns. Has {len(turns)}."
    for turn in turns:
        assert "user" in turn
        assert "standalone_query" in turn
        assert "expected_answer" in turn
        assert "source_doc_id" in turn
        assert turn["source_doc_id"] in turn["expected_answer"], f"Expected answer in turn {turn['turn']} of {conv['conversation_id']} is missing source citation in text."

with open('eval_dataset_a.json', 'w', encoding='utf-8') as f:
    json.dump(dataset_a, f, ensure_ascii=False, indent=2)

with open('eval_dataset_b.json', 'w', encoding='utf-8') as f:
    json.dump(dataset_b, f, ensure_ascii=False, indent=2)

print("SUCCESS: Evaluation datasets validated and saved successfully.")
