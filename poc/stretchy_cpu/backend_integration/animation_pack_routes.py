# Extracted integration reference from live Kid Studio backend.
# Requires: build_animation_pack_plan, get_kids_asset,
# _kids_reference_attachments, _capture_specialist, add_file,
# create_kids_asset, router, settings, FastAPI dependencies.

@app.get('/v1/kids/assets/{id}/animation-pack/plan')
async def kids_asset_animation_pack_plan(
    id:str,
    animation_profile:str='humanoid',
    account_id:str=Depends(require_account),
):
    asset=await get_kids_asset(account_id,id)

    if not asset:
        raise HTTPException(404,'Asset not found')

    if str(asset.get('type') or '').strip().lower()!='character':
        raise HTTPException(
            422,
            'Animation packs can only be created from character assets.',
        )

    if asset.get('approved') is not True:
        raise HTTPException(
            409,
            'Approve the canonical character before preparing its animation pack.',
        )

    data=asset.get('data') or {}
    files=[
        f for f in (data.get('files') or [])
        if isinstance(f,dict)
        and str(f.get('mime_type') or '').startswith('image/')
        and f.get('id')
    ]

    if not files:
        raise HTTPException(
            409,
            'The approved canonical character has no usable image file.',
        )

    reference_file_ids=[str(f['id']) for f in files]

    try:
        slots=build_animation_pack_plan(animation_profile)
    except ValueError as exc:
        raise HTTPException(422,str(exc))

    jobs=[]

    for slot in slots:
        jobs.append({
            **slot,
            'canonical_asset_id':asset['id'],
            'canonical_asset_name':asset.get('name'),
            'reference_file_ids':reference_file_ids,
            'generation_mode':'atomic_single_image',
        })

    return {
        'canonical_asset_id':asset['id'],
        'canonical_asset_name':asset.get('name'),
        'animation_profile':animation_profile,
        'reference_file_ids':reference_file_ids,
        'job_count':len(jobs),
        'required_job_count':sum(1 for j in jobs if j.get('required')),
        'jobs':jobs,
        'paid_generation_performed':False,
    }


@app.post('/v1/kids/assets/{id}/animation-pack/generate/{slot_key}')
async def kids_asset_animation_pack_generate(
    id:str,
    slot_key:str,
    animation_profile:str='humanoid',
    account_id:str=Depends(require_account),
):
    canonical=await get_kids_asset(account_id,id)

    if not canonical:
        raise HTTPException(404,'Canonical character asset not found')

    if str(canonical.get('type') or '').strip().lower()!='character':
        raise HTTPException(
            422,
            'Animation packs can only be generated from character assets.',
        )

    if canonical.get('approved') is not True:
        raise HTTPException(
            409,
            'Approve the canonical character before generating animation assets.',
        )

    canonical_data=canonical.get('data') or {}
    canonical_files=[
        f for f in (canonical_data.get('files') or [])
        if isinstance(f,dict)
        and str(f.get('mime_type') or '').startswith('image/')
        and f.get('id')
    ]

    if not canonical_files:
        raise HTTPException(
            409,
            'The approved canonical character has no usable image file.',
        )

    try:
        plan=build_animation_pack_plan(animation_profile)
    except ValueError as exc:
        raise HTTPException(422,str(exc))

    slot=next(
        (
            item for item in plan
            if str(item.get('slot_key') or '')==slot_key
        ),
        None,
    )

    if not slot:
        raise HTTPException(
            404,
            f'Animation pack slot not found: {slot_key}',
        )

    reference_file_ids=[
        str(f['id'])
        for f in canonical_files
    ]

    try:
        reference_attachments=await _kids_reference_attachments(
            account_id,
            reference_file_ids,
            'image/',
        )
    except Exception as exc:
        raise HTTPException(409,str(exc))

    generation_prompt=str(
        slot.get('prompt_suffix') or ''
    ).strip()

    if not generation_prompt:
        raise HTTPException(
            409,
            'Animation pack slot has no generation prompt.',
        )

    provider=router.choose(
        'image',
        generation_prompt,
        [],
    )

    try:
        text,arts=await _capture_specialist(
            provider,
            generation_prompt,
            'image',
            {'quality':'balanced'},
            reference_attachments,
        )
    except Exception as exc:
        raise HTTPException(
            502,
            f'Image generation failed: {exc}',
        )

    files=[]

    for art in arts:
        path=Path(art.get('path') or '')

        if not path.is_file():
            continue

        fid=str(uuid.uuid4())

        account_dir=(
            settings.upload_path
            / re.sub(
                r'[^A-Za-z0-9._-]+',
                '_',
                account_id,
            )
        )
        account_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        stored=account_dir/f'{fid}-{path.name}'
        shutil.copy2(path,stored)

        size=stored.stat().st_size
        mime=(
            mimetypes.guess_type(path.name)[0]
            or 'application/octet-stream'
        )

        await add_file(
            account_id,
            fid,
            stored.name,
            path.name,
            mime,
            size,
            str(stored),
            None,
        )

        files.append({
            'id':fid,
            'name':path.name,
            'url':f'/api/files/{fid}/download',
            'mime_type':mime,
            'size':size,
        })

    if not files:
        raise HTTPException(
            502,
            'Image provider returned no usable media file.',
        )

    asset_name=(
        f"{canonical.get('name') or 'character'}"
        f"__animation__{slot_key}"
    )

    aid=await create_kids_asset(
        account_id,
        canonical['series_id'],
        slot.get('asset_type') or 'character_animation_view',
        asset_name,
        {
            'animation_pack':True,
            'animation_profile':animation_profile,
            'slot_key':slot_key,
            'view_type':slot.get('view_type'),
            'pose':slot.get('pose'),
            'expression':slot.get('expression'),
            'required':bool(slot.get('required')),
            'canonical_asset_id':canonical['id'],
            'canonical_asset_name':canonical.get('name'),
            'reference_file_ids':reference_file_ids,
            'generation_prompt':generation_prompt,
            'prompt':generation_prompt,
            'specialist':'image',
            'provider':provider.id,
            'provider_text':text,
            'files':files,
            'generation_mode':'atomic_single_image',
        },
        canonical['series_id'],
        canonical['id'],
        'derived',
    )

    return {
        'asset_id':aid,
        'canonical_asset_id':canonical['id'],
        'slot_key':slot_key,
        'animation_profile':animation_profile,
        'asset_type':slot.get('asset_type'),
        'files':files,
        'provider':provider.id,
        'approved':False,
        'review_required':True,
    }
