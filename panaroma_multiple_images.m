clear all; close all; clc;


%% CONFIGURATION
NUM_SURF_FEATURES = 1200;
FRAME_SKIP = 2;
RANSAC_THRESHOLD = 3.0;
MIN_INLIERS = 6;

%% AUTO-DETECT DATASETS
fprintf('Auto-detecting image files...\n\n');

mov3_found = (isfile('mov3/mov3_1.jpg') || isfile('mov3_1.jpg'));
mov2_found = (isfile('mov2b_images/mov2b_7.jpg') || isfile('mov2b_7.jpg'));

if mov3_found, fprintf(' Found mov3 (indoor office)\n'); end
if mov2_found, fprintf(' Found mov2b (outdoor location)\n'); end
fprintf('\n');

%% PROCESS mov2b
if mov2_found
    fprintf('PROCESSING: mov2b Panorama\n');
    
    frame_idx = 7:FRAME_SKIP:27;
    fprintf('Frames: %s\n\n', mat2str(frame_idx));
    
    pano_mov2 = panorama_main(frame_idx, 'mov2b', NUM_SURF_FEATURES, RANSAC_THRESHOLD, MIN_INLIERS);
    
    if ~isempty(pano_mov2)
        imwrite(uint8(pano_mov2 * 255), 'panorama_mov2b_.jpg');
        fprintf('\n SAVED: panorama_mov2b_.jpg (%d × %d)\n\n', size(pano_mov2,2), size(pano_mov2,1));
        
        figure('Name', 'mov2b', 'Position', [100 550 1400 500]);
        imshow(pano_mov2);
        title(sprintf('mov2b Panorama (%d × %d)', size(pano_mov2,2), size(pano_mov2,1)));
    end
end


%% PROCESS mov3
if mov3_found
    fprintf('PROCESSING: mov3 Panorama\n');
    
    frame_idx = 1:FRAME_SKIP:17;
    fprintf('Frames: %s\n\n', mat2str(frame_idx));
    
    pano_mov3 = panorama_main(frame_idx, 'mov3', NUM_SURF_FEATURES, RANSAC_THRESHOLD, MIN_INLIERS);
    
    if ~isempty(pano_mov3)
        imwrite(uint8(pano_mov3 * 255), 'panorama_mov3_.jpg');
        fprintf('\n SAVED: panorama_mov3_.jpg (%d × %d)\n\n', size(pano_mov3,2), size(pano_mov3,1));
        
        figure('Name', 'mov3', 'Position', [100 100 1400 500]);
        imshow(pano_mov3);
        title(sprintf('mov3 Panorama (%d × %d)', size(pano_mov3,2), size(pano_mov3,1)));
    end
end

fprintf('\n PANORAMA STITCHING IS COMPLETE\n\n');


function panorama = panorama_main(frame_indices, seq_prefix, num_surf, ransac_t, min_inl)
    fprintf('STEP 1: Loading images for the process\n');
    images = {};
    num_frames = length(frame_indices);
    
    for i = 1:num_frames
        fn = frame_indices(i);
        
        if strcmp(seq_prefix, 'mov2b')
            % mov2b images are in mov2b_images/ folder
            paths = {
                sprintf('mov2b_images/mov2b_%d.jpg', fn);
                sprintf('mov2b_%d.jpg', fn)
            };
        else
            % mov3 images are in mov3/ folder
            paths = {
                sprintf('%s/%s_%d.jpg', seq_prefix, seq_prefix, fn);
                sprintf('%s_%d.jpg', seq_prefix, fn)
            };
        end
        
        found = false;
        for p = 1:length(paths)
            if isfile(paths{p})
                img = imread(paths{p});
                img_d = double(img) / 255;
                if size(img_d, 3) == 1
                    img_d = repmat(img_d, [1, 1, 3]);
                end
                images{end+1} = img_d;
                fprintf('  [%2d/%2d] Frame %2d loaded from: %s\n', i, num_frames, fn, paths{p});
                found = true;
                break;
            end
        end
        if ~found
            fprintf('  [%2d/%2d] Frame %2d NOT FOUND (tried: %s, %s)\n', i, num_frames, fn, paths{1}, paths{2});
        end
    end
    
    if length(images) < 2
        fprintf('\n  ERROR: Need at least 2 images, found %d\n', length(images));
        panorama = [];
        return;
    end
    
    [h, w, ~] = size(images{1});
    num_images = length(images);
    
    fprintf('\nSTEP 2: SURF feature detection...\n');
    keypts = {};
    descs = {};
    
    for i = 1:num_images
        gray = rgb2gray(uint8(images{i} * 255));
        pts = detectSURFFeatures(gray);
        pts = selectStrongest(pts, num_surf);
        [feat, vpts] = extractFeatures(gray, pts);
        keypts{i} = vpts.Location;
        descs{i} = feat;
        fprintf('  Frame %d: %d features\n', i, size(feat, 1));
    end
    
    fprintf('\nSTEP 3: Homography computation...\n');
    tforms = cell(num_images, 1);
    for i = 1:num_images
        tforms{i} = eye(3);
    end
    
    for n = 2:num_images
        fprintf('  Pair %d-%d: ', n-1, n);
        
        idx_pairs = matchFeatures(descs{n}, descs{n-1}, 'Unique', true);
        if size(idx_pairs, 1) < 4
            fprintf('Few matches\n');
            continue;
        end
        
        fprintf('%d matches, ', size(idx_pairs, 1));
        
        mp1 = keypts{n}(idx_pairs(:, 1), :);
        mp2 = keypts{n-1}(idx_pairs(:, 2), :);
        
        [H, n_inl] = dlt_ransac(mp1, mp2, ransac_t);
        
        if isempty(H) || n_inl < min_inl
            fprintf('RANSAC failed\n');
            continue;
        end
        
        tforms{n} = H * tforms{n-1};
        fprintf('%d inliers \n', n_inl);
    end
    
    fprintf('\nSTEP 4: Center image adjustment...\n');
    xlims = zeros(num_images, 2);
    ylims = zeros(num_images, 2);
    
    for i = 1:num_images
        [xlims(i,:), ylims(i,:)] = compute_bounds(tforms{i}, h, w);
    end
    
    avg_x = mean(xlims, 2);
    [~, idx] = sort(avg_x);
    c_idx = idx(ceil(num_images / 2));
    fprintf('  Center: %d of %d\n', c_idx, num_images);
    
    try
        T_inv = inv(tforms{c_idx});
        for i = 1:num_images
            tforms{i} = T_inv * tforms{i};
        end
    catch
    end
    
    for i = 1:num_images
        [xlims(i,:), ylims(i,:)] = compute_bounds(tforms{i}, h, w);
    end
    
    x_min = min(xlims(:, 1));
    x_max = max(xlims(:, 2));
    y_min = min(ylims(:, 1));
    y_max = max(ylims(:, 2));
    
    pano_w = round(x_max - x_min);
    pano_h = round(y_max - y_min);
    
    if pano_w > 10000 || pano_h > 5000
        scale = min(10000/pano_w, 5000/pano_h);
        sm = [scale, 0, 0; 0, scale, 0; 0, 0, 1];
        for i = 1:num_images
            tforms{i} = sm * tforms{i};
        end
        for i = 1:num_images
            [xlims(i,:), ylims(i,:)] = compute_bounds(tforms{i}, h, w);
        end
        x_min = min(xlims(:, 1));
        x_max = max(xlims(:, 2));
        y_min = min(ylims(:, 1));
        y_max = max(ylims(:, 2));
        pano_w = round(x_max - x_min);
        pano_h = round(y_max - y_min);
    end
    
    fprintf('  Size: %d × %d\n', pano_w, pano_h);
    
    fprintf('\nSTEP 5: Warping and blending\n');
    panorama = zeros(pano_h, pano_w, 3);
    wmap = zeros(pano_h, pano_w);
    
    for i = 1:num_images
        fprintf('  Image %d/%d... ', i, num_images);
        
        try
            H_inv = inv(tforms{i});
        catch
            fprintf('skip\n');
            continue;
        end
        
        for y = 1:pano_h
            for x = 1:pano_w
                wx = x_min + (x - 1);
                wy = y_min + (y - 1);
                
                src = H_inv * [wx; wy; 1];
                
                if abs(src(3)) > 1e-8
                    sx = src(1) / src(3);
                    sy = src(2) / src(3);
                    
                    if sx >= 1 && sx <= w && sy >= 1 && sy <= h
                        pix = bilinear_interp(images{i}, sx, sy);
                        wt = 1.0;
                        panorama(y, x, 1) = panorama(y, x, 1) + pix(1) * wt;
                        panorama(y, x, 2) = panorama(y, x, 2) + pix(2) * wt;
                        panorama(y, x, 3) = panorama(y, x, 3) + pix(3) * wt;
                        wmap(y, x) = wmap(y, x) + wt;
                    end
                end
            end
        end
        fprintf('✓\n');
    end
    
    for c = 1:3
        panorama(:,:,c) = panorama(:,:,c) ./ max(wmap, 1e-8);
    end
    panorama = max(0, min(1, panorama));
    
    fprintf('\nSTEP 6: Crop borders\n');
    pg = mean(panorama, 3);
    mask = pg > 0.01;
    rows = sum(mask, 2) > 0;
    cols = sum(mask, 1) > 0;
    
    ri = find(rows);
    ci = find(cols);
    
    if ~isempty(ri) && ~isempty(ci)
        panorama = panorama(ri(1):ri(end), ci(1):ci(end), :);
    end
    
    fprintf('  Final: %d × %d\n', size(panorama, 2), size(panorama, 1));
end

%% DLT + RANSAC
function [H_best, n_inl] = dlt_ransac(pc, pp, th)
    np = size(pc, 1);
    bH = [];
    mx = 0;
    
    for trial = 1:2000
        idx = randperm(np, 4);
        Ht = dlt_homography(pc(idx, :), pp(idx, :));
        
        if isempty(Ht), continue; end
        
        mask = compute_inliers(pc, pp, Ht, th);
        ni = sum(mask);
        
        if ni > mx
            mx = ni;
            bH = Ht;
            bm = mask;
        end
    end
    
    if mx >= 4
        Hr = dlt_homography(pc(bm, :), pp(bm, :));
        if ~isempty(Hr)
            bH = Hr;
        end
    end
    
    H_best = bH;
    n_inl = mx;
end

function H = dlt_homography(p1, p2)
    if size(p1, 1) < 4
        H = [];
        return;
    end
    
    n = size(p1, 1);
    [pn1, T1] = normalize_pts(p1);
    [pn2, T2] = normalize_pts(p2);
    
    A = zeros(2*n, 9);
    for i = 1:n
        x = pn1(i, 1);
        y = pn1(i, 2);
        xp = pn2(i, 1);
        yp = pn2(i, 2);
        
        A(2*i-1, :) = [-x, -y, -1, 0, 0, 0, x*xp, y*xp, xp];
        A(2*i, :) = [0, 0, 0, -x, -y, -1, x*yp, y*yp, yp];
    end
    
    try
        [~, ~, V] = svd(A);
        h = V(:, 9);
        Hn = reshape(h, 3, 3)';
        H = T2 \ (Hn * T1);
        H = H / H(3, 3);
    catch
        H = [];
    end
end

function [pn, T] = normalize_pts(p)
    c = mean(p, 1);
    d = sqrt(sum((p - c).^2, 2));
    md = mean(d);
    
    s = (md > 1e-8) * sqrt(2) / max(md, 1e-8) + (md <= 1e-8);
    T = [s, 0, -s*c(1); 0, s, -s*c(2); 0, 0, 1];
    
    ph = [p, ones(size(p,1), 1)]';
    pnh = T * ph;
    pn = pnh(1:2, :)';
end

function mask = compute_inliers(p1, p2, H, th)
    np = size(p1, 1);
    mask = false(np, 1);
    
    try
        p1h = [p1, ones(np, 1)]';
        p2ph = H * p1h;
        p2p = p2ph(1:2, :) ./ p2ph(3, :);
        err = sqrt(sum((p2' - p2p).^2, 1))';
        mask = err < th;
    catch
    end
end

%% HELPERS
function [xl, yl] = compute_bounds(H, h, w)
    c = [1, 1; w, 1; w, h; 1, h];
    ch = [c, ones(4,1)]';
    
    try
        th = H * ch;
        t = th(1:2,:) ./ th(3,:);
        xl = [min(t(1,:)), max(t(1,:))];
        yl = [min(t(2,:)), max(t(2,:))];
    catch
        xl = [1, w];
        yl = [1, h];
    end
end

function pix = bilinear_interp(img, x, y)
    [h, w, c] = size(img);
    
    x0 = floor(x);
    y0 = floor(y);
    dx = x - x0;
    dy = y - y0;
    
    x0 = max(1, min(w, x0));
    x1 = min(w, x0 + 1);
    y0 = max(1, min(h, y0));
    y1 = min(h, y0 + 1);
    
    pix = zeros(1, c);
    for ch = 1:c
        v00 = img(y0, x0, ch);
        v10 = img(y0, x1, ch);
        v01 = img(y1, x0, ch);
        v11 = img(y1, x1, ch);
        
        v0 = (1-dx)*v00 + dx*v10;
        v1 = (1-dx)*v01 + dx*v11;
        pix(ch) = (1-dy)*v0 + dy*v1;
    end
end