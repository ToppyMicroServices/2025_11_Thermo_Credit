#import <AppKit/AppKit.h>
#import <Foundation/Foundation.h>
#import <PDFKit/PDFKit.h>

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc != 4) {
            fprintf(stderr, "usage: render_pdf_pdfkit INPUT.pdf OUTPUT_DIR DPI\n");
            return 2;
        }
        NSString *inputPath = [NSString stringWithUTF8String:argv[1]];
        NSString *outputPath = [NSString stringWithUTF8String:argv[2]];
        double dpi = atof(argv[3]);
        if (dpi <= 0) {
            fprintf(stderr, "DPI must be positive\n");
            return 2;
        }
        PDFDocument *document = [[PDFDocument alloc] initWithURL:[NSURL fileURLWithPath:inputPath]];
        if (document == nil) {
            fprintf(stderr, "PDFKit could not open the input PDF\n");
            return 1;
        }
        NSError *directoryError = nil;
        [[NSFileManager defaultManager] createDirectoryAtPath:outputPath
                                  withIntermediateDirectories:YES
                                                   attributes:nil
                                                        error:&directoryError];
        if (directoryError != nil) {
            fprintf(stderr, "%s\n", directoryError.localizedDescription.UTF8String);
            return 1;
        }
        double scale = dpi / 72.0;
        for (NSInteger index = 0; index < document.pageCount; index++) {
            PDFPage *page = [document pageAtIndex:index];
            if (page == nil) {
                fprintf(stderr, "PDFKit could not read page %ld\n", (long)index + 1);
                return 1;
            }
            NSRect bounds = [page boundsForBox:kPDFDisplayBoxMediaBox];
            NSSize size = NSMakeSize(bounds.size.width * scale, bounds.size.height * scale);
            NSImage *thumbnail = [page thumbnailOfSize:size forBox:kPDFDisplayBoxMediaBox];
            NSBitmapImageRep *bitmap = [[NSBitmapImageRep alloc] initWithData:thumbnail.TIFFRepresentation];
            NSData *png = [bitmap representationUsingType:NSBitmapImageFileTypePNG properties:@{}];
            if (png == nil) {
                fprintf(stderr, "PDFKit could not encode page %ld\n", (long)index + 1);
                return 1;
            }
            NSString *name = [NSString stringWithFormat:@"page-%03ld.png", (long)index + 1];
            NSString *path = [outputPath stringByAppendingPathComponent:name];
            NSError *writeError = nil;
            [png writeToFile:path options:NSDataWritingAtomic error:&writeError];
            if (writeError != nil) {
                fprintf(stderr, "%s\n", writeError.localizedDescription.UTF8String);
                return 1;
            }
        }
        printf("PDFKit rendered %ld pages at %.0f DPI\n", (long)document.pageCount, dpi);
    }
    return 0;
}
