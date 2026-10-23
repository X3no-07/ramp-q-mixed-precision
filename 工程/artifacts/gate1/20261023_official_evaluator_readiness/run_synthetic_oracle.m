function run_synthetic_oracle(outputDir)
% Synthetic in-memory input only. Never invoke evalDET or saveAnnoRes.
% Upstream dropObjectsInIgr/evalRes/calcAccuracy are not edited or replaced.
here = fileparts(mfilename('fullpath'));
projectRoot = fileparts(fileparts(fileparts(here)));
sourceRoot = fullfile(projectRoot, 'artifacts', 'gate1', '20261022_split_refinement', ...
    'evaluation_sources', 'official_toolkit');
addpath(fullfile(sourceRoot, 'utils'), '-begin');
assert(exist(outputDir, 'dir') == 7, 'Launcher must create a new output directory');
assert(exist(fullfile(outputDir, 'oracle_results.mat'), 'file') == 0, 'Refusing existing output');
assert(exist('jsondecode', 'builtin') || exist('jsondecode', 'file'), 'jsondecode is required');
assert(exist('jsonencode', 'builtin') || exist('jsonencode', 'file'), 'jsonencode is required');
assert(exist('mean2', 'builtin') || exist('mean2', 'file'), ...
    'Native calcAccuracy requires mean2; install/load decisions are outside this runner');
required = {'dropObjectsInIgr', 'createIntImg', 'evalRes', 'compOas', 'calcAccuracy', 'VOCap'};
resolved = struct();
for n = 1:numel(required)
    resolved.(required{n}) = which(required{n});
    assert(strcmp(resolved.(required{n}), fullfile(sourceRoot, 'utils', [required{n} '.m'])), ...
        ['Unexpected function shadowing: ' required{n}]);
end
fixture = jsondecode(fileread(fullfile(here, 'synthetic_fixtures.json')));
assert(strcmp(fixture.data_role, 'synthetic'));
cases = fixture.cases;
results = cell(numel(cases), 1);
failed = 0;
for k = 1:numel(cases)
    if iscell(cases), c = cases{k}; else, c = cases(k); end
    record = struct('id', c.id, 'kind', c.kind, 'purpose', c.purpose);
    try
        images = c.images;
        allgt = cell(numel(images), 1); alldet = cell(numel(images), 1);
        intermediates = cell(numel(images), 1);
        for j = 1:numel(images)
            if iscell(images), image = images{j}; else, image = images(j); end
            rawgt = rows8(image.raw_ground_truth);
            rawdt = rows8(image.raw_predictions);
            [newgt, newdt] = dropObjectsInIgr(rawgt, rawdt, image.height, image.width);
            gt = newgt;
            gt(newgt(:, 5) == 0, 5) = 1;
            gt(newgt(:, 5) == 1, 5) = 0;
            allgt{j} = gt; alldet{j} = newdt;
            intermediate = struct('filtered_gt', newgt, 'filtered_predictions', newdt, 'prepared_gt', gt);
            if strcmp(c.kind, 'match')
                byClass = cell(10, 1);
                for category = 1:10
                    gt0 = gt(gt(:, 6) == category, 1:5);
                    dt0 = newdt(newdt(:, 6) == category, 1:5);
                    [gt1, dt1] = evalRes(gt0, dt0, c.threshold);
                    byClass{category} = struct('category', category, 'gt_output', gt1, 'dt_output', dt1);
                end
                intermediate.class_matches = byClass;
            end
            intermediates{j} = intermediate;
        end
        record.intermediates = intermediates;
        if strcmp(c.kind, 'aggregate')
            [AP_all, AP_50, AP_75, AR_1, AR_10, AR_100, AR_500] = calcAccuracy(numel(images), allgt, alldet);
            record.metrics = struct('AP_all', AP_all, 'AP_50', AP_50, 'AP_75', AP_75, ...
                'AR_1', AR_1, 'AR_10', AR_10, 'AR_100', AR_100, 'AR_500', AR_500);
        end
        record.status = 'EXECUTED_NATIVE_SOURCE';
    catch err
        record.status = 'NATIVE_CASE_ERROR';
        record.error_identifier = err.identifier;
        record.error_message = err.message;
        failed = failed + 1;
    end
    results{k} = record;
end
receipt = struct('schema', 'rampq.gate1.synthetic_official_oracle_execution/v1', ...
    'runtime', version, 'computer', computer, 'resolved_functions', resolved, ...
    'fixture_count', numel(cases), 'case_errors', failed, ...
    'real_data_read', false, 'model_executed', false, 'official_AP_equivalence_certified', false, ...
    'scope', 'Native synthetic outputs only; no comparison against a full Python AP implementation exists');
% MAT preserves empty dimensions and NaN values that JSON may normalize to null.
save(fullfile(outputDir, 'oracle_results.mat'), 'results', 'receipt', '-v7');
fid = fopen(fullfile(outputDir, 'oracle_results.json'), 'w');
assert(fid >= 0); cleaner = onCleanup(@() fclose(fid));
fwrite(fid, jsonencode(struct('receipt', receipt, 'results', {results})), 'char');
clear cleaner;
assert(failed == 0, 'One or more synthetic native cases failed; inspect saved outputs');
end

function matrix = rows8(value)
if isempty(value)
    matrix = zeros(0, 8);
else
    matrix = double(value);
    assert(size(matrix, 2) == 8, 'Fixture matrices require eight columns');
    assert(all(isfinite(matrix(:))), 'Nonfinite fixture input');
end
end
