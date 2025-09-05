import unittest


class TestPrepareForPrompt(unittest.TestCase):
    def test_prepare_for_promt(self):
        from src.citation.cication_manager import CitationManager

        metadatas = [
            {"source": "source1", "url": "url1", "favicon": "favicon1"},
            {"source": "source2", "url": "url2", "favicon": "favicon2"},
            {"source": "source3", "url": "url3", "favicon": "favicon3"},
            {"source": "source4", "url": "url4", "favicon": "favicon4"},
            {"source": "source5", "url": "url5", "favicon": "favicon5"},
            {"source": "source6", "url": "url6", "favicon": "favicon6"},
            {"source": "source7", "url": "url7", "favicon": "favicon7"},
            {"source": "source8", "url": "url8", "favicon": "favicon8"},
        ]

        texts = [
            "text1",
            "text2",
            "text3",
            "text4",
            "text5",
            "text6",
            "text7",
            "text8",
        ]

        citation_manager = CitationManager(metadatas=metadatas)

        text_prompt = citation_manager.prepare_for_prompt(
            texts=texts, metadatas=metadatas
        )

        expected_text_prompt = (
            "[0] text1\n"
            "[1] text2\n"
            "[2] text3\n"
            "[3] text4\n"
            "[4] text5\n"
            "[5] text6\n"
            "[6] text7\n"
            "[7] text8"
        )

        self.assertEqual(text_prompt, expected_text_prompt)

        text_prompt = citation_manager.prepare_for_prompt(
            texts=texts[1:], metadatas=metadatas[1:]
        )

        expected_text_prompt = (
            "[1] text2\n"
            "[2] text3\n"
            "[3] text4\n"
            "[4] text5\n"
            "[5] text6\n"
            "[6] text7\n"
            "[7] text8"
        )

        self.assertEqual(text_prompt, expected_text_prompt)


import unittest

from src.citation.cication_manager import CitationManager


class TestCitation(unittest.TestCase):
    def run_test(self, metadatas, original_texts, expected_reorder_texts, expected_ids):
        original_texts = (
            [original_texts] if isinstance(original_texts, str) else original_texts
        )
        expected_reorder_texts = (
            [expected_reorder_texts]
            if isinstance(expected_reorder_texts, str)
            else expected_reorder_texts
        )

        citation_manager = CitationManager(metadatas=metadatas)
        reordered_texts, ids = citation_manager.reorder_citations(original_texts.copy())

        self.assertEqual(reordered_texts, expected_reorder_texts)
        self.assertEqual(ids, expected_ids)

        for original_text, reordered_text in zip(original_texts, reordered_texts):
            original_ids = citation_manager.retrieve_ids_from_text(original_text)
            reordered_ids = citation_manager.retrieve_ids_from_text(reordered_text)
            original_urls = [
                metadatas[id]["url"] for id in original_ids if id < len(metadatas)
            ]
            reordered_urls = [
                citation_manager.id2metadata[id].url for id in reordered_ids
            ]

            self.assertEqual(set(original_urls), set(reordered_urls))

    def test_basic_reordering(self):
        metadatas = [
            {"source": f"source{i}", "url": f"url{i}", "favicon": f"favicon{i}"}
            for i in range(1, 9)
        ]
        original_text = (
            "This is a text [0] [1] [2]\n"
            "This is a text [3] [4] [5]\n\n"
            "This is a text [6] [7] [0]\n"
        )
        expected_reorder_text = (
            "This is a text [1] [2] [3]\n"
            "This is a text [4] [5] [6]\n\n"
            "This is a text [7] [8] [1]\n"
        )
        expected_ids = [1, 2, 3, 4, 5, 6, 7, 8]
        self.run_test(metadatas, original_text, expected_reorder_text, expected_ids)

    def test_mixed_order_citations(self):
        metadatas = [
            {"source": f"source{i}", "url": f"url{i}", "favicon": f"favicon{i}"}
            for i in range(20)
        ]
        original_text = "Text [5] more [2] text [18] and [5] again [0] with [19] repeats [2] end [5]"
        expected_reorder_text = (
            "Text [1] more [2] text [3] and [1] again [4] with [5] repeats [2] end [1]"
        )
        expected_ids = [1, 2, 3, 4, 5]
        self.run_test(metadatas, original_text, expected_reorder_text, expected_ids)

    def test_citations_with_gaps(self):
        metadatas = [
            {"source": f"source{i}", "url": f"url{i}", "favicon": f"favicon{i}"}
            for i in range(30)
        ]
        original_text = (
            "Start [25] then [10] gap [0] and [29] out [15] of [5] order [20] citations"
        )
        expected_reorder_text = (
            "Start [1] then [2] gap [3] and [4] out [5] of [6] order [7] citations"
        )
        expected_ids = [1, 2, 3, 4, 5, 6, 7]
        self.run_test(metadatas, original_text, expected_reorder_text, expected_ids)

    def test_citations_with_numbers_in_text(self):
        metadatas = [
            {"source": f"source{i}", "url": f"url{i}", "favicon": f"favicon{i}"}
            for i in range(15)
        ]
        original_text = "Edge [14]case[0]no[1]space[13]and[2]numbers[12]like 42 [3] or [11][10][9] end"
        expected_reorder_text = (
            "Edge [1]case[2]no[3]space[4]and[5]numbers[6]like 42 [7] or [8][9][10] end"
        )
        expected_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.run_test(metadatas, original_text, expected_reorder_text, expected_ids)

    def test_citations_with_unicode(self):
        metadatas = [
            {"source": f"source{i}", "url": f"url{i}", "favicon": f"favicon{i}"}
            for i in range(10)
        ]
        original_text = (
            "Unicode: [3]🌟[2], punctuation: [1]![4]?[0].[5]; and [7]([6])[8] mix [9]"
        )
        expected_reorder_text = (
            "Unicode: [1]🌟[2], punctuation: [3]![4]?[5].[6]; and [7]([8])[9] mix [10]"
        )
        expected_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.run_test(metadatas, original_text, expected_reorder_text, expected_ids)

    def test_long_text_with_many_citations(self):
        metadatas = [
            {"source": f"source{i}", "url": f"url{i}", "favicon": f"favicon{i}"}
            for i in range(100)
        ]
        original_text = " ".join(
            [f"[{i}] Word" for i in range(0, 100, 2)]
            + [f"[{i}] Another" for i in range(1, 100, 2)]
        )
        expected_reorder_text = " ".join(
            [f"[{i}] Word" for i in range(1, 51)]
            + [f"[{i}] Another" for i in range(51, 101)]
        )
        expected_ids = list(range(1, 101))
        self.run_test(metadatas, original_text, expected_reorder_text, expected_ids)

    def test_multiple_texts_with_overlapping_citations(self):
        metadatas = [
            {"source": f"source{i}", "url": f"url{i}", "favicon": f"favicon{i}"}
            for i in range(10)
        ]
        original_texts = [
            "First text [3] [1] [5] [2]",
            "Second text [4] [2] [6] [1]",
            "Third text [7] [5] [3] [8]",
        ]
        expected_reorder_texts = [
            "First text [1] [2] [3] [4]",
            "Second text [5] [4] [6] [2]",
            "Third text [7] [3] [1] [8]",
        ]
        expected_ids = [1, 2, 3, 4, 5, 6, 7, 8]
        self.run_test(metadatas, original_texts, expected_reorder_texts, expected_ids)

    def test_texts_with_invalid_ids(self):
        metadatas = [
            {"source": f"source{i}", "url": f"url{i}", "favicon": f"favicon{i}"}
            for i in range(5)
        ]
        original_texts = [
            "Valid and invalid [2] [10] [1] [15]",
            "More invalid [20] [3] [25] [4]",
        ]
        expected_reorder_texts = ["Valid and invalid [1] [2]", "More invalid [3] [4]"]
        expected_ids = [1, 2, 3, 4]
        self.run_test(metadatas, original_texts, expected_reorder_texts, expected_ids)

    def test_complex_ordering_with_invalid_ids(self):
        metadatas = [
            {"source": f"source{i}", "url": f"url{i}", "favicon": f"favicon{i}"}
            for i in range(15)
        ]
        original_texts = [
            "First [5] then [2] and [10] with [7]",
            "Invalid [20] then valid [8] [1] [15]",
            "Mixed [12] [3] [25] [6] [5] [9]",
        ]
        expected_reorder_texts = [
            "First [1] then [2] and [3] with [4]",
            "Invalid then valid [5] [6]",
            "Mixed [7] [8] [9] [1] [10]",
        ]
        expected_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.run_test(metadatas, original_texts, expected_reorder_texts, expected_ids)

    def test_order_preservation(self):
        metadatas = [
            {"source": f"source{i}", "url": f"url{i}", "favicon": f"favicon{i}"}
            for i in range(10)
        ]
        original_texts = [
            "Reverse order [9] [8] [7] [6]",
            "Mixed order [2] [5] [1] [3]",
        ]
        expected_reorder_texts = [
            "Reverse order [1] [2] [3] [4]",
            "Mixed order [5] [6] [7] [8]",
        ]
        expected_ids = [1, 2, 3, 4, 5, 6, 7, 8]
        self.run_test(metadatas, original_texts, expected_reorder_texts, expected_ids)

    def test_whitespace_preservation(self):
        metadatas = [
            {"source": f"source{i}", "url": f"url{i}", "favicon": f"favicon{i}"}
            for i in range(15)
        ]
        original_texts = [
            "First [5] then [2] and [10] with [7]",
            "Invalid [20] then valid [8][1][15]",
            "Mixed [12][3][25][6][5][9]",
        ]
        expected_reorder_texts = [
            "First [1] then [2] and [3] with [4]",
            "Invalid then valid [5][6]",
            "Mixed [7][8][9][1][10]",
        ]
        expected_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.run_test(metadatas, original_texts, expected_reorder_texts, expected_ids)

    def test_different_citation_formats(self):
        metadatas = [
            {"source": f"source{i}", "url": f"url{i}", "favicon": f"favicon{i}"}
            for i in range(15)
        ]
        original_texts = [
            "First [5] then [2] and [10] with [7]",
            "Invalid [20] then valid [8,1,15]",
            "Mixed [12,3,25][6][5,9]",
        ]
        expected_reorder_texts = [
            "First [1] then [2] and [3] with [4]",
            "Invalid then valid [5,6]",
            "Mixed [7,8][9][1,10]",
        ]
        expected_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.run_test(metadatas, original_texts, expected_reorder_texts, expected_ids)

    def test_large_gaps_between_citation_numbers(self):
        metadatas = [
            {"source": f"source{i}", "url": f"url{i}", "favicon": f"favicon{i}"}
            for i in range(1000)
        ]
        original_texts = ["Large gaps [1] [500] [999]", "More gaps [250] [750] [2]"]
        expected_reorder_texts = ["Large gaps [1] [2] [3]", "More gaps [4] [5] [6]"]
        expected_ids = [1, 2, 3, 4, 5, 6]
        self.run_test(metadatas, original_texts, expected_reorder_texts, expected_ids)

    def test_citations_with_leading_zeros(self):
        metadatas = [
            {"source": f"source{i}", "url": f"url{i}", "favicon": f"favicon{i}"}
            for i in range(20)
        ]
        original_texts = [
            "Leading zeros [01] [05] [10]",
            "More leading zeros [015] [02] [7]",
        ]
        expected_reorder_texts = [
            "Leading zeros [1] [2] [3]",
            "More leading zeros [4] [5] [6]",
        ]
        expected_ids = [1, 2, 3, 4, 5, 6]
        self.run_test(metadatas, original_texts, expected_reorder_texts, expected_ids)

    def test_duplicate_citations_and_close_numbers(self):
        metadatas = [
            {"source": f"source{i}", "url": f"url{i}", "favicon": f"favicon{i}"}
            for i in range(15)
        ]
        original_texts = [
            "Duplicates [1] [1] [2] [2] [3]",
            "Close numbers [10] [11] [12] [13] [1]",
        ]
        expected_reorder_texts = [
            "Duplicates [1] [1] [2] [2] [3]",
            "Close numbers [4] [5] [6] [7] [1]",
        ]
        expected_ids = [1, 2, 3, 4, 5, 6, 7]
        self.run_test(metadatas, original_texts, expected_reorder_texts, expected_ids)

    def test_citations_at_word_boundaries(self):
        metadatas = [
            {"source": f"source{i}", "url": f"url{i}", "favicon": f"favicon{i}"}
            for i in range(10)
        ]
        original_texts = [
            "Word[1]boundaries[2]and[3]punctuation",
            "More[4]examples:[5],[6];[7]![8]?[9]",
        ]
        expected_reorder_texts = [
            "Word[1]boundaries[2]and[3]punctuation",
            "More[4]examples:[5],[6];[7]![8]?[9]",
        ]
        expected_ids = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.run_test(metadatas, original_texts, expected_reorder_texts, expected_ids)

    def test_citations_with_varying_digit_count(self):
        metadatas = [
            {"source": f"source{i}", "url": f"url{i}", "favicon": f"favicon{i}"}
            for i in range(1000)
        ]
        original_texts = [
            "Varying digits [1] [10] [100]",
            "More variation [500] [50] [5] [999]",
        ]
        expected_reorder_texts = [
            "Varying digits [1] [2] [3]",
            "More variation [4] [5] [6] [7]",
        ]
        expected_ids = [1, 2, 3, 4, 5, 6, 7]
        self.run_test(metadatas, original_texts, expected_reorder_texts, expected_ids)


class TestCitationManager(unittest.TestCase):
    """
    Auto generated with Claude Sonnet
    """

    def setUp(self):
        from src.citation.cication_manager import CitationManager

        self.cm = CitationManager()
        self.cm.add_metadatas(
            [
                {
                    "source": "Source1",
                    "url": "http://example1.com",
                    "favicon": "favicon1.ico",
                },
                {
                    "source": "Source2",
                    "url": "http://example2.com",
                    "favicon": "favicon2.ico",
                },
                {
                    "source": "Source3",
                    "url": "http://example3.com",
                    "favicon": "favicon3.ico",
                },
                {
                    "source": "Source4",
                    "url": "http://example4.com",
                    "favicon": "favicon4.ico",
                },
            ]
        )

    def test_reorder_citations_basic(self):
        texts = [
            "This is a test [2,1]. Another citation [3].",
            "Here's another text with [0,2] citations.",
        ]
        reordered_texts, cited_ids = self.cm.reorder_citations(texts)

        self.assertEqual(
            reordered_texts,
            [
                "This is a test [1,2]. Another citation [3].",
                "Here's another text with [4,1] citations.",
            ],
        )
        self.assertEqual(cited_ids, [1, 2, 3, 4])

    def test_reorder_citations_missing_ids(self):
        texts = ["This citation [5] doesn't exist.", "This one [2] does exist."]
        reordered_texts, cited_ids = self.cm.reorder_citations(texts)

        self.assertEqual(
            reordered_texts,
            ["This citation doesn't exist.", "This one [1] does exist."],
        )
        self.assertEqual(cited_ids, [1])

    def test_reorder_citations_empty_brackets(self):
        texts = ["This is an empty citation [].", "This is a valid citation [3]."]
        reordered_texts, cited_ids = self.cm.reorder_citations(texts)

        self.assertEqual(
            reordered_texts,
            ["This is an empty citation [].", "This is a valid citation [1]."],
        )
        self.assertEqual(cited_ids, [1])

    def test_reorder_citations_preserve_newlines(self):
        texts = ["Line 1 [1]\nLine 2 [2]\nLine 3 [3]"]
        reordered_texts, cited_ids = self.cm.reorder_citations(texts)

        self.assertEqual(reordered_texts, ["Line 1 [1]\nLine 2 [2]\nLine 3 [3]"])
        self.assertEqual(cited_ids, [1, 2, 3])

    def test_reorder_citations_multiple_citations_per_line(self):
        texts = ["Multiple citations [2,1] in one line [4,3]."]
        reordered_texts, cited_ids = self.cm.reorder_citations(texts)

        self.assertEqual(reordered_texts, ["Multiple citations [1,2] in one line [3]."])
        self.assertEqual(cited_ids, [1, 2, 3])

    def test_reorder_citations_update_internal_mappings(self):
        texts = ["Reorder all citations [4,3,2,1]."]
        self.cm.reorder_citations(texts)

        # Check if internal mappings are updated
        self.assertEqual(
            self.cm.url2id,
            {
                "http://example1.com": 4,
                "http://example2.com": 3,
                "http://example3.com": 2,
                "http://example4.com": 1,
            },
        )
        self.assertEqual(list(self.cm.id2metadata.keys()), [4, 3, 2, 1])

    def test_reorder_citations_unused_ids(self):
        texts = ["Only use some citations [2,1]."]
        reordered_texts, cited_ids = self.cm.reorder_citations(texts)

        self.assertEqual(reordered_texts, ["Only use some citations [1,2]."])
        self.assertEqual(cited_ids, [1, 2])

        # Check if unused IDs are still in the mappings
        self.assertEqual(len(self.cm.id2metadata), 4)
        self.assertEqual(len(self.cm.url2id), 4)
