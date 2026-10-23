function run_full_ap_stress_oracle(outputDir)
% New synthetic input suite only; official source functions are never changed.
here = fileparts(mfilename('fullpath'));
root = fileparts(fileparts(fileparts(here)));
sourceRoot = fullfile(root, 'artifacts', 'gate1', '20261022_split_refinement', ...
    'evaluation_sources', 'official_toolkit');
addpath(fullfile(sourceRoot, 'utils'), '-begin');
assert(exist(outputDir, 'dir') == 7, 'Launcher must create new output directory');
assert(exist(fullfile(outputDir, 'oracle_results.mat'), 'file') == 0, 'Refusing existing output');
assert(exist(fullfile(outputDir, 'oracle_results.json'), 'file') == 0, 'Refusing existing output');
assert(exist('mean2', 'builtin') || exist('mean2', 'file'), 'Use the real image package mean2');
required = {'dropObjectsInIgr', 'createIntImg', 'evalRes', 'compOas', 'calcAccuracy', 'VOCap'};
resolved = struct();
for n = 1:numel(required)
    resolved.(required{n}) = which(required{n});
    assert(strcmp(resolved.(required{n}), fullfile(sourceRoot, 'utils', [required{n} '.m'])), ...
        ['Unexpected official function shadowing: ' required{n}]);
end
fixture = jsondecode(fileread(fullfile(here, 'stress_fixtures.json')));
assert(strcmp(fixture.data_role, 'synthetic') && fixture.no_real_data_or_model_access);
assert(numel(fixture.cases) == 12);
cases = fixture.cases;
results = cell(numel(cases), 1);
failed = 0;
for k = 1:numel(cases)
    if iscell(cases), c = cases{k}; else, c = cases(k); end
    assert(strcmp(c.kind, 'aggregate'));
    record = struct('id', c.id, 'kind', c.kind, 'purpose', c.purpose);
    try
        images = c.images;
        allgt = cell(numel(images), 1); alldet = cell(numel(images), 1);
        intermediates = cell(numel(images), 1);
        for j = 1:numel(images)
            if iscell(images), image = images{j}; else, image = images(j); end
            [newgt, newdt] = dropObjectsInIgr(rows8(image.raw_ground_truth), ...
                rows8(image.raw_predictions), image.height, image.width);
            gt = newgt;
            gt(newgt(:,5) == 0, 5) = 1;
            gt(newgt(:,5) == 1, 5) = 0;
            allgt{j} = gt; alldet{j} = newdt;
            intermediates{j} = struct('filtered_gt', newgt, 'filtered_predictions', newdt, 'prepared_gt', gt);
        end
        record.intermediates = intermediates;
        [AP_all, AP_50, AP_75, AR_1, AR_10, AR_100, AR_500] = calcAccuracy(numel(images), allgt, alldet);
        record.metrics = struct('AP_all', AP_all, 'AP_50', AP_50, 'AP_75', AP_75, ...
            'AR_1', AR_1, 'AR_10', AR_10, 'AR_100', AR_100, 'AR_500', AR_500);
        record.status = 'EXECUTED_NATIVE_SOURCE';
    catch err
        record.status = 'NATIVE_CASE_ERROR';
        record.error_identifier = err.identifier;
        record.error_message = err.message;
        failed = failed + 1;
    end
    results{k} = record;
end
receipt = struct('schema', 'rampq.gate1.synthetic_official_stress_oracle_execution/v1', ...
    'runtime', version, 'computer', computer, 'resolved_functions', resolved, ...
    'mean2_resolved_path', which('mean2'), 'fixture_count', numel(cases), 'case_errors', failed, ...
    'real_data_read', false, 'model_executed', false, 'official_AP_equivalence_certified', false, ...
    'selection_accessed', false, 'holdout_accessed', false, 'official_sources_edited', false, ...
    'scope', 'Declared synthetic stress outputs only; no real-data or arbitrary-input AP equivalence');
save(fullfile(outputDir, 'oracle_results.mat'), 'results', 'receipt', '-v7');
fid = fopen(fullfile(outputDir, 'oracle_results.json'), 'w');
assert(fid >= 0); cleaner = onCleanup(@() fclose(fid));
fwrite(fid, jsonencode(struct('receipt', receipt, 'results', {results})), 'char');
clear cleaner;
assert(failed == 0, 'One or more stress cases failed; inspect saved outputs');
end

function matrix = rows8(value)
if isempty(value)
    matrix = zeros(0, 8);
else
    matrix = double(value);
    assert(size(matrix, 2) == 8 && all(isfinite(matrix(:))), 'Finite eight-column input required');
end
end
