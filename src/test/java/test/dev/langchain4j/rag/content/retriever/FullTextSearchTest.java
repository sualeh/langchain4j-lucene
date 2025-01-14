package test.dev.langchain4j.rag.content.retriever;

import static org.hamcrest.CoreMatchers.is;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.hamcrest.Matchers.hasSize;

import dev.langchain4j.data.document.Metadata;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.rag.content.Content;
import dev.langchain4j.rag.content.retriever.lucene.DirectoryFactory;
import dev.langchain4j.rag.content.retriever.lucene.LuceneContentRetriever;
import dev.langchain4j.rag.content.retriever.lucene.LuceneIndexer;
import dev.langchain4j.rag.query.Query;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;
import org.apache.lucene.store.Directory;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

public class FullTextSearchTest {

    private static final TextSegment[] hitTextSegments = {
        TextSegment.from("Lucene is a powerful search library.", metadataName("doc1")),
        TextSegment.from("Is Lucinity is not a library?", metadataName("doc2")),
        TextSegment.from("BM25Similarity is the default search term similarity algorithm.", metadataName("doc3")),
    };
    private static final TextSegment[] missTextSegments = {
        TextSegment.from("Aaaaah - some random text here.", metadataName("miss1"))
    };
    private static final Query query = Query.from("Give me information on the lucine search library");

    private static Metadata metadataName(final String name) {
        final Metadata metadata = new Metadata();
        metadata.put("name", name);
        return metadata;
    }

    private Directory directory;
    private LuceneIndexer indexer;
    private LuceneContentRetriever contentRetriever;

    @Test
    public void query1() {

        contentRetriever = LuceneContentRetriever.builder()
                .topNMatches(1)
                .directory(directory)
                .build();

        for (final TextSegment textSegment : hitTextSegments) {
            indexer.addContent(textSegment);
        }
        final List<String> expectedTextSegments = new ArrayList<>();
        expectedTextSegments.add(hitTextSegments[2].text());

        final List<Content> results = contentRetriever.retrieve(query);
        final List<String> actualTextSegments =
                results.stream().map(content -> content.textSegment().text()).collect(Collectors.toList());
        Collections.sort(actualTextSegments);

        assertThat(results, hasSize(1));
        assertThat(actualTextSegments, is(expectedTextSegments));
    }

    @Test
    public void queryAll() {

        contentRetriever = LuceneContentRetriever.builder()
                .matchUntilTopN()
                .directory(directory)
                .build();

        final List<String> expectedTextSegments = new ArrayList<>();
        for (final TextSegment textSegment : hitTextSegments) {
            indexer.addContent(textSegment);
            expectedTextSegments.add(textSegment.text());
        }
        for (final TextSegment textSegment : missTextSegments) {
            indexer.addContent(textSegment);
            expectedTextSegments.add(textSegment.text());
        }
        Collections.sort(expectedTextSegments);

        final List<Content> results = contentRetriever.retrieve(query);
        final List<String> actualTextSegments =
                results.stream().map(content -> content.textSegment().text()).collect(Collectors.toList());
        Collections.sort(actualTextSegments);

        assertThat(results, hasSize(hitTextSegments.length + missTextSegments.length));
        assertThat(actualTextSegments, is(expectedTextSegments));
    }

    @Test
    public void queryContent() {

        contentRetriever = LuceneContentRetriever.builder().directory(directory).build();

        final List<String> expectedTextSegments = new ArrayList<>();
        for (final TextSegment textSegment : hitTextSegments) {
            indexer.addContent(textSegment);
            expectedTextSegments.add(textSegment.text());
        }
        for (final TextSegment textSegment : missTextSegments) {
            indexer.addContent(textSegment);
        }
        Collections.sort(expectedTextSegments);

        final List<Content> results = contentRetriever.retrieve(query);
        final List<String> actualTextSegments =
                results.stream().map(content -> content.textSegment().text()).collect(Collectors.toList());
        Collections.sort(actualTextSegments);

        assertThat(results, hasSize(hitTextSegments.length));
        assertThat(actualTextSegments, is(expectedTextSegments));
    }

    @Test
    public void queryWithMaxTokens() {

        contentRetriever = LuceneContentRetriever.builder()
                .maxTokens(8)
                .directory(directory)
                .build();

        for (final TextSegment textSegment : hitTextSegments) {
            indexer.addContent(textSegment);
        }
        final List<String> expectedTextSegments = new ArrayList<>();
        expectedTextSegments.add(hitTextSegments[0].text());

        final List<Content> results = contentRetriever.retrieve(query);
        final List<String> actualTextSegments =
                results.stream().map(content -> content.textSegment().text()).collect(Collectors.toList());
        Collections.sort(actualTextSegments);

        assertThat(results, hasSize(1));
        assertThat(actualTextSegments, is(expectedTextSegments));
    }

    @BeforeEach
    public void setUp() {
        directory = DirectoryFactory.tempDirectory();
        indexer = new LuceneIndexer(directory);
    }

    @AfterEach
    public void tearDown() throws Exception {
        directory.close();
    }
}
