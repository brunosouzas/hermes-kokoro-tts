import pytest

import kokoro_language as lang

VOICES = {"pt": "pf_dora", "en": "bf_emma"}


@pytest.mark.parametrize("text,expected", [
    ("Hi Bruno. Your deployment finished and the pipeline is green.", "en"),
    ("What is next on the board today?", "en"),
    ("Olá, Bruno. O deploy terminou e o pipeline está verde.", "pt"),
    ("Hoje não tem nada marcado na sua agenda.", "pt"),
    ("Você pode revisar isso depois?", "pt"),
    ("Ação concluída", "pt"),  # accents alone are a Portuguese signal
    ("MuleSoft Anypoint", None),  # names only: no signal
    ("12:30", None),
    ("", None),
    # Review regressions: shared words, loanword accents, URLs and code must not decide.
    ("Me avisa quando terminar.", "pt"),
    ("Me avisa.", None),
    ("Please update my résumé.", "en"),
    ("The café is open.", "en"),
    ("Se for possível, me liga.", "pt"),
    ("Leia https://example.com/in/it/the/docs agora.", "pt"),
    ("Rode `git push origin main` depois do merge.", "pt"),
    ("Run `npm install o pacote` and then restart it.", "en"),
    ("The era of local models is here.", "en"),
])
def test_detect(text, expected):
    assert lang.detect(text) == expected


# Italian (issue #5): the sentences from the report, plus the kind of thing Hermes says day to day.
ITALIAN = [
    "Ciao, come stai?",
    "Oggi il tempo è bello.",
    "Vorrei un caffè, per favore.",
    "Il progetto è quasi finito.",
    "Non lo so, forse domani.",
    "Abbiamo bisogno di più tempo.",
    "Oggi il tempo è bello e vogliamo uscire a camminare nel parco.",
    "Il deploy è terminato e la pipeline è verde.",
    "Hai una riunione alle tre con il team.",
    "Ho controllato la posta: non ci sono messaggi urgenti.",
    "La build è fallita, ma ho già trovato il problema.",
    "Vuoi che ti ricordi della chiamata più tardi?",
    "Questa settimana hai tre scadenze.",
    "Ecco il riepilogo della giornata.",
    "Il server è di nuovo online.",
    "Però non è ancora pronto.",
    "Grazie, a dopo!",
    "Nessun evento in calendario per oggi.",
    "L'aggiornamento è disponibile.",
    "L’aggiornamento è disponibile.",  # typographic apostrophe
    "C'è un nuovo messaggio da Marco.",
    "Così va bene.",
    "Sono già le nove e mezza.",
    "Tutto a posto, niente di nuovo.",
    "Il meeting è stato spostato a giovedì.",
    "Perché no?",
    "Un'altra volta.",  # the elision alone decides
    "Nell’ufficio.",
    "Giovedì mattina.",  # the grave accent alone decides
    "Poiché è tardi, riprovo domani.",  # acute accents of Italian words do not count for Portuguese
    "Poiché non funziona, riprovo.",
    "Benché sia tardi, lavoro ancora.",
    "Finché non arriva, aspetto.",
    "Va bene, lo faccio subito.",
]


@pytest.mark.parametrize("text", ITALIAN)
def test_detect_italian(text):
    assert lang.detect(text) == "it"


@pytest.mark.parametrize("text,expected", [
    # Words Italian shares with English or Portuguese must not pull those away from their language.
    ("I have one per day.", "en"),
    ("Come in, the door is open.", "en"),
    ("This is a non-blocking change.", "en"),
    ("Non-interactive mode.", None),  # was None before Italian, must not become Italian
    ("Plain chat (non-slash messages) is unaffected.", "en"),
    ("Reject non-`SELECT` queries.", None),
    ("Set use_non_blocking_copy.", None),
    ("Partner access allows 30 days STO.", None),
    ("NVIDIA Ampere or AMD MI200.", "en"),
    ("Auditoria local àquele cliente.", "pt"),
    # Italian-looking words that are everyday Portuguese.
    ("Estou com sono.", "pt"),
    ("Que sono!", "pt"),
    ("Ele mora no Rio Grande do Sul.", "pt"),
    ("Moro na zona sul.", None),
    ("Preciso de um prego.", "pt"),
    ("Legal, né?", "pt"),
    ("Tá bom, né?", "pt"),
    ("Copo d'água.", "pt"),
    ("Refiro-me àquilo.", "pt"),
    # Acronyms are not Italian words.
    ("Use DI for this.", "en"),
    ("Ask MA about it.", "en"),
    ("All's well that ends well.", "en"),
    ("The pasta is called penne alla vodka.", "en"),
    ("Eu sei quando ele chega.", "pt"),
    ("Vou à reunião às três.", "pt"),  # the crase is still Portuguese
    ("Vou fazê-lo amanhã.", "pt"),  # the enclitic -lo is not the Italian article
    ("Posso ver a lei depois?", "pt"),
    ("Ele deve chegar daqui a pouco.", "pt"),
    ("A tua casa é linda.", "pt"),
])
def test_italian_words_shared_with_other_languages_do_not_move_them(text, expected):
    assert lang.detect(text) == expected


def test_grave_a_is_not_a_portuguese_accent_when_italian_is_scored():
    assert lang.detect("Città bellissima.") is None
    assert lang.detect("Città bellissima.", ("en", "pt")) == "pt"  # as before Italian existed


def test_italian_is_scored_only_when_it_has_a_voice():
    assert lang.languages_for({"pt": "pf_dora", "en": "bf_emma"}) == ("en", "pt")
    assert lang.languages_for({"it": "im_nicola"}) == ("en", "pt", "it")
    # With an en/pt map, an Italian-looking sentence keeps the previous voice instead of
    # switching to the default one.
    text = "The build passed. Mia sent a message."
    assert lang.plan(text, VOICES, "af_heart") == [(text, "bf_emma")]
    text = "O deploy terminou. Legal, né?"
    assert lang.plan(text, VOICES, "af_heart") == [(text, "pf_dora")]


def test_italian_voice_is_picked_per_sentence():
    voices = {"it": "im_nicola", "en": "af_heart"}
    text = "Il deploy è terminato. The tests passed."
    assert lang.plan(text, voices, "af_heart") == [
        ("Il deploy è terminato.", "im_nicola"),
        ("The tests passed.", "af_heart"),
    ]


def test_supported_languages():
    assert set(lang.SUPPORTED) == {"en", "pt", "it"}


def test_split_sentences():
    assert lang.split_sentences("One. Two!  Three?\nFour") == ["One.", "Two!", "Three?", "Four"]


@pytest.mark.parametrize("text,expected", [
    ("O Dr. Brown me ligou hoje. Depois eu volto.", ["O Dr. Brown me ligou hoje.", "Depois eu volto."]),
    ("Mr. Smith is here. He is waiting.", ["Mr. Smith is here.", "He is waiting."]),
    ("Use e.g. the default. Then restart.", ["Use e.g. the default.", "Then restart."]),
    ("Talk to J. Smith today. Thanks.", ["Talk to J. Smith today.", "Thanks."]),
    ("A versão 4.9 saiu. Atualize.", ["A versão 4.9 saiu.", "Atualize."]),
    ("Veja example.com. Depois volte.", ["Veja example.com.", "Depois volte."]),
    ("Ele disse \"pronto.\" Then he left.", ["Ele disse \"pronto.\"", "Then he left."]),
    ("- primeiro item\n- second item", ["- primeiro item", "- second item"]),
    ("Fim de frase com é. Depois outra.", ["Fim de frase com é.", "Depois outra."]),
])
def test_split_sentences_edge_cases(text, expected):
    assert lang.split_sentences(text) == expected


def test_punctuation_inside_inline_code_neither_splits_nor_switches():
    text = "Leia `the and. is are were` agora. Depois volte."
    assert lang.split_sentences(text) == ["Leia `the and. is are were` agora.", "Depois volte."]
    assert lang.plan(text, VOICES, "pf_dora") == [(text, "pf_dora")]


def test_abbreviation_does_not_switch_voice_mid_sentence():
    assert lang.plan("O Dr. Brown me ligou hoje.", VOICES, "pf_dora") == [("O Dr. Brown me ligou hoje.", "pf_dora")]


def test_plan_switches_voice_per_sentence_and_merges_neighbours():
    text = "Olá, Bruno. O deploy terminou. The pipeline is green and the tests passed. Até amanhã!"
    assert lang.plan(text, VOICES, "pf_dora") == [
        ("Olá, Bruno. O deploy terminou.", "pf_dora"),
        ("The pipeline is green and the tests passed.", "bf_emma"),
        ("Até amanhã!", "pf_dora"),
    ]


def test_unclear_sentence_keeps_previous_voice():
    text = "The build is done. MuleSoft 4.9. It is deployed."
    assert lang.plan(text, VOICES, "pf_dora") == [("The build is done. MuleSoft 4.9. It is deployed.", "bf_emma")]


def test_first_unclear_sentence_uses_default_voice():
    assert lang.plan("Anypoint.", VOICES, "pf_dora") == [("Anypoint.", "pf_dora")]


def test_unmapped_language_uses_default_voice():
    assert lang.plan("The pipeline is green.", {"pt": "pf_dora"}, "pf_dora") == [("The pipeline is green.", "pf_dora")]


def test_whitespace_only_text_is_returned_as_is():
    assert lang.plan("   ", VOICES, "pf_dora") == [("   ", "pf_dora")]
