package dev.langchain4j.rag.content.retriever.lucene;

import static java.util.Objects.requireNonNull;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.MMapDirectory;

public class DirectoryFactory {

    /**
     * Create a memory mapped file-system based directory.
     *
     * @param directoryPath Path for the directory. It should not previously exist.
     * @return
     */
    public static Directory fsDirectory(final Path directoryPath) {
        requireNonNull(directoryPath, "No file system directory path provided");
        try {
            final Directory directory = new MMapDirectory(directoryPath);
            return directory;
        } catch (final Exception e) {
            throw new RuntimeException(e.getMessage(), e);
        }
    }

    public static Directory tempDirectory() {
        try {
            final Path directoryPath = Files.createTempDirectory(Directory.class.getCanonicalName());
            final Path newSubDirectory = Paths.get(directoryPath.toString(), Directory.class.getCanonicalName());
            return fsDirectory(newSubDirectory);
        } catch (final Exception e) {
            throw new RuntimeException(e.getMessage(), e);
        }
    }

    private DirectoryFactory() {
        // Prevent instantiation
    }
}
